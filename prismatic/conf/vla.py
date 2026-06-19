"""PoLAR VLA training configurations."""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Optional, Union

from draccus import ChoiceRegistry


@dataclass
class VLAConfig(ChoiceRegistry):
    # fmt: off
    vla_id: str
    base_vlm: Union[str, Path]
    freeze_vision_backbone: bool
    freeze_llm_backbone: bool
    unfreeze_last_llm_layer: bool

    data_mix: str
    shuffle_buffer_size: int

    epochs: int
    max_steps: Optional[int]

    expected_world_size: int
    global_batch_size: int
    per_device_batch_size: int

    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    lr_scheduler_type: str
    warmup_ratio: float

    train_strategy: str
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision_training: bool = True
    reduce_in_full_precision: bool = True
    # fmt: on


@dataclass
class Exp_DinoSigLIP_224px_Bridge(VLAConfig):
    vla_id: str = "prism-dinosiglip-224px+mx-bridge"
    base_vlm: Union[str, Path] = "prism-dinosiglip-224px+7b"

    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = True

    data_mix: str = "bridge"
    shuffle_buffer_size: int = 20_000

    epochs: int = 10
    max_steps: Optional[int] = None

    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32

    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0

    train_strategy: str = "fsdp-full-shard"


@unique
class VLARegistry(Enum):
    DINOSIGLIP_224PX_MX_BRIDGE = Exp_DinoSigLIP_224px_Bridge

    @property
    def vla_id(self) -> str:
        return self.value.vla_id


for vla_variant in VLARegistry:
    VLAConfig.register_subclass(vla_variant.vla_id, vla_variant.value)
