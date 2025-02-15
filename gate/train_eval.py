import os
import pathlib
from typing import List, Optional

import hydra

import pytorch_lightning
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.tuner.tuning import Tuner
from rich.traceback import install
from wandb.util import generate_id

from gate.base.utils.loggers import get_logger
from gate.configs import get_module_import_path
from gate.configs.callbacks import LogConfigInformation
from gate.datamodules.base import DataModule
from gate.train_eval_agents.base import TrainingEvaluationAgent

log = get_logger(__name__)

install(show_locals=False, word_wrap=True, width=350)


def checkpoint_setup(config):
    checkpoint_path = None

    if config.resume:

        log.info("Continue from existing checkpoint")

        if not pathlib.Path(f"{config.current_experiment_dir}").exists():
            os.makedirs(f"{config.current_experiment_dir}", exist_ok=True)

        checkpoint_path = f"{config.current_experiment_dir}/checkpoints/last.ckpt"

        if not pathlib.Path(checkpoint_path).exists():
            checkpoint_path = None

        log.info(checkpoint_path)

    else:

        log.info("Starting from scratch")
        if not pathlib.Path(f"{config.current_experiment_dir}").exists():
            os.makedirs(f"{config.current_experiment_dir}", exist_ok=True)

    return checkpoint_path


def train_eval(config: DictConfig):
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    if config.get("seed"):
        seed_everything(config.seed, workers=True)
    # --------------------------------------------------------------------------------
    # Create or recover checkpoint path to resume from
    checkpoint_path = checkpoint_setup(config)
    # --------------------------------------------------------------------------------
    # Instantiate Lightning DataModule for task
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    # log information regarding data module to be instantiated -- particularly the class name that is stored in _target_
    datamodule: DataModule = hydra.utils.instantiate(
        config.datamodule, _recursive_=False
    )
    # List in comments all possible datamodules/datamodule configs
    datamodule.setup(stage="fit")
    # datamodule_pretty_dict_tree = generate_config_tree(
    #     config=datamodule.__dict__, resolve=True
    # )
    log.info(
        f"Datamodule <{config.datamodule._target_}> instantiated, "
        f"with attributes {datamodule.__dict__}"
    )
    # --------------------------------------------------------------------------------
    # Instantiate Lightning TrainingEvaluationAgent for task
    log.info(f"Instantiating model <{config.model._target_}>")

    train_eval_agent: TrainingEvaluationAgent = hydra.utils.instantiate(
        config.train_eval_agent, datamodule=datamodule, _recursive_=False
    )
    # --------------------------------------------------------------------------------
    # Instantiate Lightning Learner using a dummy data dict with the
    # data names and shapes
    x_dummy_data_dict, y_dummy_data_dict = datamodule.dummy_batch()

    # depth first traversal and printing of tensor shapes
    _ = train_eval_agent.forward((x_dummy_data_dict, y_dummy_data_dict))
    # --------------------------------------------------------------------------------
    # Instantiate Lightning Callbacks
    # --------------------------------------------------------------------------------
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                if (
                    cb_conf["_target_"].split(".")[-1]
                    == get_module_import_path(LogConfigInformation).split(".")[-1]
                ):
                    log.info(
                        f"Instantiating config collection callback <{cb_conf._target_}>"
                    )
                    cb_conf["config_dict"] = OmegaConf.to_container(
                        config, resolve=True
                    )
                    callbacks.append(
                        hydra.utils.instantiate(
                            cb_conf,
                            _recursive_=False,
                        )
                    )
                else:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(hydra.utils.instantiate(cb_conf))

    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = generate_id()
    # --------------------------------------------------------------------------------
    # Instantiate Experiment Logger
    # --------------------------------------------------------------------------------
    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # --------------------------------------------------------------------------------
    # Instantiate Lightning Trainer
    # --------------------------------------------------------------------------------
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
    )

    # --------------------------------------------------------------------------------
    # If auto_scale_batch_size is set, we need to tune the batch size using
    # the Lightning Tuner class, starting from given batch size and increasing
    # in powers of 2
    if config.trainer.auto_scale_batch_size:
        tuner = Tuner(trainer)
        new_batch_size = tuner.scale_batch_size(
            train_eval_agent,
            datamodule=datamodule,
            mode="power",
            init_val=2 * torch.cuda.device_count(),
        )
        datamodule.batch_size = new_batch_size
        config.datamodule.batch_size = new_batch_size

    # --------------------------------------------------------------------------------
    # Start training
    if config.mode.fit:
        log.info("Starting training!")
        trainer.validate(
            model=train_eval_agent,
            datamodule=datamodule,
            ckpt_path=checkpoint_path,
        )

        trainer.fit(
            model=train_eval_agent,
            datamodule=datamodule,
            ckpt_path=checkpoint_path,
        )

    # --------------------------------------------------------------------------------
    # Start evaluation on test set
    if config.mode.test and not config.trainer.get("fast_dev_run"):
        datamodule.setup(stage="test")

        log.info("Starting testing ! 🧪")

        if config.mode.fit is False:
            test_results = trainer.test(
                model=train_eval_agent,
                datamodule=datamodule,
                verbose=False,
                ckpt_path=checkpoint_path,
            )
        else:
            test_results = trainer.test(
                model=train_eval_agent,
                datamodule=datamodule,
                verbose=False,
            )

        log.info(
            f"Testing results can be found in the wandb log: {wandb.run.url}, "
            f"please only check that when finalizing your conclusions, "
            f"otherwise you are at risk of subconsciosuly biasing your "
            f"results 🚨"
        )
        for logger_instance in logger:
            if isinstance(logger_instance, pytorch_lightning.loggers.wandb.WandbLogger):
                wandb.log(test_results[0], step=0)
    # Make sure everything closed properly
    log.info("Finalizing! 😺")
    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    wandb.finish(quiet=False)
