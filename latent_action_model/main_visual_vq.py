import sys
from pathlib import Path

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

from lightning.pytorch.cli import LightningCLI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genie.dataset import LightningOpenX
from genie.model_visual_vq import VisualVQ_DINO_LAM


cli = LightningCLI(
    VisualVQ_DINO_LAM,
    LightningOpenX,
    seed_everything_default=42,
    save_config_kwargs={"overwrite": True},
)
