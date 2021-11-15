import argparse
import glob
import os

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import WandbLogger

from gate.datasets import load_dataset
from gate.datasets.dataset_loading_hub import add_dataset_args
from gate.models import model_library_dict
from gate.utils.arg_parsing import process_args
from gate.utils.logging_helpers import get_logging
from gate.utils.storage import build_experiment_folder


def get_base_argument_parser():
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("--general.experiment_name", type=str, default="dev")
    parser.add_argument("--general.seed", type=int, default=0)
    parser.add_argument(
        "--general.logger_level",
        type=str,
        default="NOSET",
        choices=["NOSET", "WARN", "INFO", "ERROR", "CRITICAL", "DEBUG"],
    )
    parser.add_argument(
        "--general.filepath_to_arguments_json_config", type=str, default=None
    )

    # dataset
    # add additional dataset specific arguments inside the dataset class
    parser.add_argument("--dataset.name", type=str, default="tali")
    parser.add_argument(
        "--dataset.data_filepath", type=str, default="data/sample_dataset"
    )
    parser.add_argument("--dataset.batch_size", type=int, default=32)
    parser.add_argument("--dataset.eval_batch_size", type=int, default=32)
    parser.add_argument("--dataset.image_height", type=int, default=224)
    parser.add_argument("--dataset.image_width", type=int, default=224)
    parser.add_argument(
        "--dataset.rescan_dataset_files", default=False, action="store_true"
    )
    parser.add_argument("--dataset.num_data_provider_workers", type=int, default=8)
    parser.add_argument("--dataset.prefetch_factor", type=int, default=2)

    # task
    # add additional task specific arguments inside the task class
    parser.add_argument("--task.name", type=str, default="tali")
    # add task specific arguments inside the task class

    # model
    # add additional model specific arguments inside the model class
    parser.add_argument("--model.name", type=str, default="ResNet")

    # logging
    parser.add_argument("--logger.offline_mode", default=False, action="store_true")
    parser.add_argument(
        "--logger.upload_model_weights", default=False, action="store_true"
    )

    parser.add_argument("--logger.logs_path", type=str, default="log")

    # adaptation schcmes
    # add additional adaptation schcmes specific arguments inside the adaptation
    # schcmes class
    parser.add_argument("--adaptation_scheme.learning_rate", type=float, default=0.001)
    parser.add_argument("--adaptation_scheme.max_epochs", type=int, default=300)
    parser.add_argument(
        "--adaptation_scheme.scheduler",
        type=str,
        default="CosineAnnealing",
        help="Scheduler for learning rate annealing: CosineAnnealing | MultiStep",
    )
    parser.add_argument(
        "--adaptation_scheme.optimizer_name",
        type=str,
        default="Adam",
        help="Optimizer?",
    )
    parser.add_argument(
        "--adaptation_scheme.min_learning_rate",
        type=float,
        default=0.0,
        help="Min learning rate the scheduler should anneal towards",
    )

    parser.add_argument("----adaptation_scheme.weight_decay", type=float, default=0)
    parser.add_argument("----adaptation_scheme.momentum", type=float, default=0.9)

    return parser


if __name__ == "__main__":
    argument_parser = get_base_argument_parser()

    argument_parser = pl.Trainer.add_argparse_args(argument_parser)
    argument_parser = LightningExperiment.add_task_specific_args(argument_parser)
    argument_parser = add_dataset_args(parser=argument_parser)
    argument_parser = model_library_dict[argument_parser.parse_args().model.name]
    args = process_args(argument_parser)
    logging = get_logging(level=args.level)

    # save_dir: Optional[str] = None,
    # offline: Optional[bool] = False,
    # id: Optional[str] = None,
    # anonymous: Optional[bool] = None,
    # version: Optional[str] = None,
    # project: Optional[str] = None,
    # log_model: Optional[bool] = False,
    # experiment = None,
    # prefix: Optional[str] = "",
    # sync_step: Optional[bool] = None,

    ############################################################################# Admin
    (saved_models_filepath, logs_filepath, images_filepath,) = build_experiment_folder(
        experiment_name=args.experiment_name, log_path=args.logs_path
    )

    ############################################################################## Data

    #  if you have an environment variable set, prioritise this over the default
    # argparse option
    environment_data_filepath = os.environ.get("PYTORCH_DATA_LOC")
    if environment_data_filepath is not None and os.path.exists(
        environment_data_filepath
    ):
        logging.warning(
            f"You have a data filepath set in your environment: "
            f"{environment_data_filepath}. This will override "
            f"argparse. "
        )
        data_filepath = environment_data_filepath
    else:
        data_filepath = args.data_filepath

    seed_everything(args.seed, workers=True)

    wandb_instance = WandbLogger(
        project="GATE",
        save_dir=f"{args.logs_path}/{args.experiment_name}/",
        offline=args.offline_mode,
        log_model=args.upload_model_weights,
    )  # your credentials

    # Always save a snapshot of the current state of the code. I've found this helps
    # immensely if you find that one of your many experiments was actually quite good
    # but you forgot what you did

    snapshot_filename = f"{saved_models_filepath}/snapshot.tar.gz"
    filetypes_to_include = [
        ".py",
        ".sh",
        ".gitignore",
        ".yml",
        ".md",
        "LICENSE",
        ".github",
    ]
    all_files = []

    for filetype in filetypes_to_include:
        all_files += glob.glob("**/*.py", recursive=True)

    for file in all_files:
        wandb_instance.experiment.log_artifact(file)

    trainer_args = pl.Trainer.parse_argparser(arg_parser=argument_parser)

    checkpoint_loss_callback = ModelCheckpoint(
        monitor="validation/overall_cross_entropy_epoch",
        dirpath=f"{args.logs_path}/{args.experiment_name}/checkpoints/",
        filename="model-loss-{epoch:02d}",
        save_on_train_epoch_end=True,
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        every_n_val_epochs=1,
        mode="min",
    )

    checkpoint_acc_callback = ModelCheckpoint(
        monitor="validation/overall_accuracy_epoch",
        dirpath=f"{args.logs_path}/{args.experiment_name}/checkpoints/",
        filename="model-accuracy-{epoch:02d}",
        save_on_train_epoch_end=True,
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        every_n_val_epochs=1,
        mode="max",
    )

    train_set, val_set, test_set, dummy_set = load_dataset(
        dataset_name=args.dataset.name,
        data_filepath=args.dataset.data_filepath,
        seed=args.general.seed,
        data_args=args.dataset,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_data_provider_workers,
        prefetch_factor=args.prefetch_factor,
    )
    dummy_batch = next(iter(dummy_set))

    experiment = LightningExperiment(
        task_args=args.task,
        model_args=args.model,
        adaptation_scheme_args=args.adaptation_scheme,
        full_args=args.args,
    )

    logging.info(
        f"max-epochs: {args.max_epochs}," f"current-epoch: {experiment.current_epoch}",
    )

    summary_callback = ModelSummary(model=experiment, max_depth=1)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    logging.info(summary_callback)

    if args.resume_from_checkpoint == "scratch":
        args.resume_from_checkpoint = (
            f"{args.logs_path}/{args.experiment_name}/checkpoints/"
        )

    trainer = pl.Trainer.from_argparse_args(
        args=trainer_args,
        logger=wandb_instance,
        log_every_n_steps=1,
        default_root_dir=f"{args.logs_path}/{args.experiment_name}/",
        resume_from_checkpoint=args.resume_from_checkpoint,
        callbacks=[
            checkpoint_loss_callback,
            checkpoint_acc_callback,
            lr_monitor,
        ],
    )

    trainer.fit(experiment, train_set_loader, val_set_loader)

    test_results = trainer.test(dataloaders=test_set_loader)
