import pytest

import torch
import torch.nn.functional as F
from dotted_dict import DottedDict

from gate.base.utils.loggers import get_logger
from gate.class_configs.base import (
    ModalitiesSupportedConfig,
    ImageClassificationTaskModuleConfig,
    ShapeConfig,
)
from gate.learners.single_layer_fine_tuning import LinearLayerFineTuningScheme
from gate.models.resnet import ImageResNet

log = get_logger(__name__, set_default_handler=True)


@pytest.mark.parametrize(
    "learner",
    [
        LinearLayerFineTuningScheme,
    ],
)
@pytest.mark.parametrize(
    "fine_tune_all_layers",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "max_epochs",
    [1, 100],
)
@pytest.mark.parametrize(
    "min_learning_rate",
    [0.00001],
)
@pytest.mark.parametrize(
    "lr",
    [0.01],
)
@pytest.mark.parametrize(
    "betas",
    [[0.9, 0.999]],
)
@pytest.mark.parametrize(
    "eps",
    [0.000001],
)
@pytest.mark.parametrize(
    "weight_decay",
    [0.00001],
)
@pytest.mark.parametrize(
    "amsgrad",
    [
        False,
        True,
    ],
)
def test_single_layer_fine_tuning(
    learner,
    fine_tune_all_layers,
    max_epochs,
    min_learning_rate,
    lr,
    betas,
    eps,
    weight_decay,
    amsgrad,
):
    task_config = ImageClassificationTaskModuleConfig(
        output_shape_dict=DottedDict({"image": DottedDict(num_classes=10)})
    )

    module = learner(
        optimizer_config=DottedDict(_target_="torch.optim.Adam", lr=0.001),
        lr_scheduler_config=DottedDict(
            _target_="torch.optim.lr_scheduler.CosineAnnealingLR",
            T_max=max_epochs,
            eta_min=0,
            verbose=True,
        ),
        fine_tune_all_layers=fine_tune_all_layers,
        max_epochs=max_epochs,
        min_learning_rate=min_learning_rate,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )
    model = ImageResNet(
        input_shape_dict=ShapeConfig(
            image=DottedDict(
                shape=DottedDict(
                    channels=3,
                    width=32,
                    height=32,
                ),
                dtype=torch.float32,
            ),
        ),
        model_name_to_download="resnet50",
        pretrained=True,
    )
    dummy_x = {
        "image": torch.randn(
            size=[
                2,
                model.input_shape_dict.image.shape.channels,
                model.input_shape_dict.image.shape.height,
                model.input_shape_dict.image.shape.width,
            ]
        )
    }

    log.info(f"dummy_x.shape: {dummy_x['image'].shape}")

    _ = model.forward(dummy_x)

    module.build(
        model=model,
        input_shape_dict=model.input_shape_dict,
        output_shape_dict=task_config.output_shape_dict,
        task_config=task_config,
        modality_config=ModalitiesSupportedConfig(image=True),
    )
    optimizer = module.configure_optimizers()["optimizer"]
    out = module.forward(dummy_x)

    loss = F.cross_entropy(out["image"], torch.randint(0, 10, (2,)))

    loss.backward()

    optimizer.step()
