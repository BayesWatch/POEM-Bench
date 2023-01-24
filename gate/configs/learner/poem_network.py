from dataclasses import MISSING, dataclass
from tkinter.tix import Tree
from typing import Any, Callable, Dict, Optional

from gate.configs import get_module_import_path
from gate.configs.learner.learning_rate_scheduler_config import (
    CosineAnnealingLRConfig,
    LRSchedulerConfig,
)
from gate.configs.learner.linear_layer_fine_tuning import LearnerConfig
from gate.configs.learner.optimizer_config import (
    AdamOptimizerConfig,
    BaseOptimizerConfig,
)
from gate.learners.poem import PartialObservationExpertsModelling
from gate.learners.protonet_poem_architecture import PrototypicalNetworkPOEMHead
from gate.learners.matchingnet_poem_architecture import MatchingNetworkPOEMHead

from gate.model_blocks.auto_builder_modules.poem_blocks import (
    HeadConv,
    HeadMLP,
    HeadResNetBlock,
)


@dataclass
class HeadConfig:
    output_activation_fn: Any = None
    view_information_num_filters: Optional[int] = None
    _target_: str = "HeadClassPath"


@dataclass
class HeadConvConfig(HeadConfig):
    num_output_filters: int = 512
    num_layers: int = 3
    num_hidden_filters: int = 512
    input_avg_pool_size: int = 7
    _target_: str = get_module_import_path(HeadConv)


@dataclass
class HeadMLPConfig(HeadConfig):
    num_output_filters: int = 512
    num_layers: int = 3
    num_hidden_filters: int = 512
    input_avg_pool_size: int = 1
    _target_: str = get_module_import_path(HeadMLP)


@dataclass
class HeadResNetBlockConfig(HeadConfig):
    num_output_filters: int = 512
    num_hidden_filters: int = 512
    _target_: str = get_module_import_path(HeadResNetBlock)


@dataclass
class PartialObservationExpertsModellingConfig(LearnerConfig):
    _target_: str = get_module_import_path(PartialObservationExpertsModelling)
    fine_tune_all_layers: bool = True
    use_input_instance_norm: bool = True
    use_mean_head: bool = True
    use_precision_head: bool = True
    head_num_layers: int = 3
    head_num_hidden_filters: int = 512
    head_num_output_filters: int = 512
    optimizer_config: BaseOptimizerConfig = AdamOptimizerConfig(lr=2e-5)
    lr_scheduler_config: LRSchedulerConfig = CosineAnnealingLRConfig()
    mean_head_config: HeadConfig = HeadMLPConfig()
    precision_head_config: HeadConfig = HeadMLPConfig()


@dataclass
class PartialObservationExpertsModellingMLPHeadConfig(
    PartialObservationExpertsModellingConfig
):
    mean_head_config: HeadConfig = HeadMLPConfig()
    precision_head_config: HeadConfig = HeadMLPConfig()


@dataclass
class PartialObservationExpertsModellingConvHeadConfig(
    PartialObservationExpertsModellingConfig
):
    mean_head_config: HeadConfig = HeadConvConfig()
    precision_head_config: HeadConfig = HeadConvConfig()


@dataclass
class PartialObservationExpertsModellingResNetHeadConfig(
    PartialObservationExpertsModellingConfig
):
    mean_head_config: HeadConfig = HeadResNetBlockConfig()
    precision_head_config: HeadConfig = HeadResNetBlockConfig()


@dataclass
class PrototypicalNetworkPOEMHeadConfig(LearnerConfig):
    _target_: str = get_module_import_path(PrototypicalNetworkPOEMHead)
    fine_tune_all_layers: bool = True
    use_input_instance_norm: bool = True
    use_mean_head: bool = True
    use_precision_head: bool = True
    head_num_layers: int = 3
    head_num_hidden_filters: int = 512
    head_num_output_filters: int = 512
    optimizer_config: BaseOptimizerConfig = AdamOptimizerConfig(lr=2e-5)
    lr_scheduler_config: LRSchedulerConfig = CosineAnnealingLRConfig()
    mean_head_config: HeadConfig = HeadMLPConfig()
    precision_head_config: HeadConfig = HeadMLPConfig()


@dataclass
class MatchingNetworkPOEMHeadConfig(LearnerConfig):
    _target_: str = get_module_import_path(MatchingNetworkPOEMHead)
    fine_tune_all_layers: bool = True
    use_input_instance_norm: bool = True
    use_mean_head: bool = True
    use_precision_head: bool = True
    head_num_layers: int = 3
    head_num_hidden_filters: int = 512
    head_num_output_filters: int = 512
    optimizer_config: BaseOptimizerConfig = AdamOptimizerConfig(lr=2e-5)
    lr_scheduler_config: LRSchedulerConfig = CosineAnnealingLRConfig()
    mean_head_config: HeadConfig = HeadMLPConfig()
    precision_head_config: HeadConfig = HeadMLPConfig()
