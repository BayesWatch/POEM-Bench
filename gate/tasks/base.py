import logging
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from gate.adaptation_schemes import adaptation_scheme_library_dict
from gate.models import model_library_dict
from gate.utils.general_utils import compute_accuracy


# TODO instead remove full args


class TaskModule(pl.LightningModule):
    def __init__(self, task_args, model_args, adaptation_scheme_args, full_args):
        super(TaskModule, self).__init__()

        self.save_hyperparameters()
        self.task_args = task_args
        self.model_args = model_args
        self.adaptation_scheme_args = adaptation_scheme_args
        self.full_args = full_args
        self.args = self.hparams

    def build(self, dummy_batch):
        raise NotImplementedError

    @staticmethod
    def add_task_specific_args(parser):
        return parser

    def forward(self, input_dict):
        raise NotImplementedError

    def collect_metrics(self, metrics_dict, phase_name):
        for metric_key, metric_value in metrics_dict.items():
            self.log(
                name=f"{phase_name}/overall_{metric_key}",
                value=metric_value,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def predict_step(
            self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError


class BaseTask(TaskModule):
    def __init__(self, task_args, model_args, adaptation_scheme_args, full_args):
        super(BaseTask, self).__init__(
            task_args, model_args, adaptation_scheme_args, full_args
        )

    @property
    def task_metrics(self):
        raise NotImplementedError

    @staticmethod
    def add_task_specific_args(parser):
        return parser

    def forward(self, input_dict):
        return self.learning_system.inference_step(input_dict)

    def collect_metrics(self, metrics_dict, phase_name):
        for metric_key, metric_value in metrics_dict.items():
            self.log(
                name=f"{phase_name}/overall_{metric_key}",
                value=metric_value,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )

    def training_step(self, batch, batch_idx):
        iter_metrics = self.learning_system.train_step(
            batch=batch, metrics=self.task_metrics
        )

        self.collect_metrics(metrics_dict=iter_metrics, phase_name="training")
        return iter_metrics["loss"]

    def validation_step(self, batch, batch_idx):
        iter_metrics = self.learning_system.evaluation_step(
            batch=batch, metrics=self.task_metrics
        )
        self.collect_metrics(metrics_dict=iter_metrics, phase_name="validation")

    def test_step(self, batch, batch_idx):
        iter_metrics = self.learning_system.evaluation_step(
            batch=batch, metrics=self.task_metrics
        )

        self.collect_metrics(metrics_dict=iter_metrics, phase_name="testing")

    def predict_step(
            self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        return self.learning_system.inference_step(batch)

    def configure_optimizers(self):
        logging.info(
            f"optimizer: {repr(self.learning_system.optimizer)},"
            f"max-epochs: {self.adaptation_scheme_args.max_epochs},"
            f"current-epoch: {self.current_epoch}",
        )
        return {
            "optimizer": self.learning_system.optimizer,
            "lr_scheduler": self.learning_system.scheduler,
        }


class ImageClassificationTask(BaseTask):
    def __init__(self, task_args, model_args, adaptation_scheme_args, full_args):
        super(ImageClassificationTask, self).__init__(
            task_args, model_args, adaptation_scheme_args, full_args
        )

    @property
    def task_metrics(self):
        return {
            "cross_entropy": lambda x, y: F.cross_entropy(input=x, target=y),
            "accuracy": lambda x, y: compute_accuracy(x, y),
        }

    def build(self, dummy_batch):
        input_dict, output_dict = dummy_batch
        self.model = model_library_dict[self.model_args.type](**self.model_args)
        self.learning_system = adaptation_scheme_library_dict[
            self.adaptation_args.type
        ](
            model=self.model,
            input_shape_dict={"image": input_dict["image"].shape[1:]},
            output_shape_dict={"image": output_dict["image"].shape[1:]},
            output_layer_activation=nn.Identity(),
            **self.adaptation_args,
        )

        self.model.build(dummy_batch)
        self.learning_system.reset_learning()
        self.learning_system.set_task_input_output_shapes(
            input_shape=self.task.input_shape_dict,
            output_shape=self.task.output_shape_dict,
        )


class ReconstructionTask(BaseTask):
    def __init__(self, task_args, model_args, adaptation_scheme_args, full_args):
        super(ReconstructionTask, self).__init__(
            task_args, model_args, adaptation_scheme_args, full_args
        )

    def task_metrics(self):
        return {
            "mse": lambda x, y: F.mse_loss(input=x, target=y),
            "mae": lambda x, y: F.l1_loss(input=x, target=y),
        }

    def build(self, dummy_batch):
        input_dict, output_dict = dummy_batch
        self.model = model_library_dict[self.model_args.type](**self.model_args)
        self.learning_system = adaptation_scheme_library_dict[
            self.adaptation_args.type
        ](
            model=self.model,
            input_shape_dict={"image": input_dict["image"].shape[1:]},
            output_shape_dict={"image": output_dict["image"].shape[1:]},
            output_layer_activation=nn.Identity(),
            **self.adaptation_args,
        )

        self.model.build(dummy_batch)
        self.learning_system.reset_learning()
        self.learning_system.set_task_input_output_shapes(
            input_shape=self.task.input_shape_dict,
            output_shape=self.task.output_shape_dict,
        )
