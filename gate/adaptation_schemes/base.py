import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


class BaseAdaptationScheme(nn.Module):
    def __init__(self):
        super(BaseAdaptationScheme, self).__init__()

    def optimizers(self, **kwargs):
        return NotImplementedError(
            f"Optimizer not implemented in model: " f"{self.__class__.__name__}"
        )

    def train_step(self, **kwargs):
        return NotImplementedError(
            f"Train step not implemented in model: " f"{self.__class__.__name__}"
        )

    def eval_step(self, **kwargs):
        return NotImplementedError(
            f"Evaluation step not implemented in model: " f"{self.__class__.__name__}"
        )


# a. Embedding component
# b. Adaptation_scheme component: Updates weights, provide a dictionary/set of which
# layers to update
# c.

# learning system: <"model" (architecture, weights), way to process different combos
# of modalities,
# way to update the model, way to transforms inputs, way to map embeddings to outputs

# learning_system.reset_learning()
# learning_system.set_task_input_output_shapes(
#     input_shape=self.input_shape,
#     output_shape=self.output_shape,
# )
# iter_metrics = learning_system.train_step(
# inputs=x, targets=y, metrics=self.metrics
# )
import logging


class ImageOnlyLinearLayerFineTuningScheme(BaseAdaptationScheme):
    def __init__(
            self,
            model,
            input_shape_dict,
            output_shape_dict,
            output_layer_activation,
            min_learning_rate=1e-6,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            amsgrad=False,
    ):
        super(ImageOnlyLinearLayerFineTuningScheme, self).__init__()
        self.model = model

        self.lr = lr
        self.min_learning_rate = min_learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.input_shape_dict = input_shape_dict
        self.output_shape_dict = output_shape_dict
        self.output_layer_activation = output_layer_activation
        self.build(input_shape_dict, output_shape_dict, output_layer_activation)
        self.optimizers()

    def setup_task_input_output_details(
            self, input_shape_dict, output_shape_dict, output_layer_activation
    ):
        self.build(input_shape_dict, output_shape_dict, output_layer_activation)

    def build(self, input_shape_dict, output_shape_dict, output_layer_activation):
        image_dummy_x = torch.randn(input_shape_dict["image"])
        model_features = self.model.forward_image(image_dummy_x)
        model_features_flatten = model_features.view((model_features.shape[0], -1))
        self.linear_output_layer = nn.Linear(
            in_features=model_features_flatten.shape[1],
            out_features=output_shape_dict["image"],
            bias=True,
        )
        logits = self.output_layer_activation(
            self.linear_output_layer(model_features_flatten)
        )

        logging.info(
            f"Built {self.__class__.__name__} "
            f"with input_shape {input_shape_dict} {image_dummy_x.shape}"
            f"with output_shape {output_shape_dict} {logits.shape}"
        )

    def reset_learning(self):
        self.linear_output_layer.reset_parameters()

    def optimizers(self):
        self.optimizer = Adam(
            params=self.linear_output_layer.parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )

        self.scheduler = CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.num_epochs,
            eta_min=self.min_learning_rate,
        )

        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def training_step(self, batch, metrics):
        input_dict, target_dict = batch
        features = self.model.forward_image(input_dict["image"])
        features_flatten = features.view((features.shape[0], -1))
        logits = self.output_layer_activation(
            self.linear_output_layer(features_flatten)
        )

        loss = F.cross_entropy(logits, target_dict["image"])

        output_dict = {
            metric_key: metric_function(logits, target_dict["image"])
            for metric_key, metric_function in metrics.items()
        }
        output_dict["loss"] = loss

        return output_dict

    def evaluation_step(self, batch, metrics):
        input_dict, target_dict = batch
        features = self.model.forward_image(input_dict["image"])
        features_flatten = features.view((features.shape[0], -1))
        logits = self.output_layer_activation(
            self.linear_output_layer(features_flatten)
        )

        return {
            metric_key: metric_function(logits, target_dict["image"])
            for metric_key, metric_function in metrics.items()
        }

    def inference_step(self, batch):
        input_dict, target_dict = batch
        features = self.model.forward_image(input_dict["image"])
        features_flatten = features.view((features.shape[0], -1))
        logits = self.output_layer_activation(
            self.linear_output_layer(features_flatten)
        )

        return features, logits
