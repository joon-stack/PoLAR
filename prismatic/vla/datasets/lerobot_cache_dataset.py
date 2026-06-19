"""
LeRobot offline window cache dataset for PoLAR finetuning.

The cache stores resized JPEG frames and JSONL windows so PoLAR training does
not need LeRobot, PyAV, pandas, or AV1 decoding in the finetune environment.
"""

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

try:
    from PIL import Image as PILImage

    _BICUBIC = PILImage.Resampling.BICUBIC
except AttributeError:  # pragma: no cover - Pillow < 9
    _BICUBIC = Image.BICUBIC


class LeRobotWindowCacheDataset(IterableDataset):
    """Iterates cached LeRobot windows in the format expected by the PoLAR window-action collator."""

    def __init__(
        self,
        cache_dir: Path,
        image_transform: Callable[[Image.Image], Any],
        image_transform_lam: Callable[[Image.Image], torch.Tensor],
        window_size: int = 10,
        shuffle_buffer_size: int = 16000,
        image_aug: bool = True,
        train: bool = True,
        dataset_name: Optional[str] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.image_transform = image_transform
        self.image_transform_lam = image_transform_lam
        self.window_size = int(window_size)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.image_aug = bool(image_aug)
        self.train = bool(train)

        metadata_path = self.cache_dir / "metadata.json"
        windows_path = self.cache_dir / "windows.jsonl"
        statistics_path = self.cache_dir / "dataset_statistics.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing LeRobot cache metadata: {metadata_path}")
        if not windows_path.exists():
            raise FileNotFoundError(f"Missing LeRobot cache windows: {windows_path}")
        if not statistics_path.exists():
            raise FileNotFoundError(f"Missing LeRobot cache statistics: {statistics_path}")

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        with open(statistics_path, "r") as f:
            self.dataset_statistics = json.load(f)

        self.dataset_name = dataset_name or self.metadata.get("dataset_name", "lerobot")
        if self.dataset_name not in self.dataset_statistics:
            raise KeyError(
                f"Dataset name {self.dataset_name!r} not found in {statistics_path}; "
                f"available={list(self.dataset_statistics)}"
            )

        action_stats = self.dataset_statistics[self.dataset_name]["action"]
        self.action_low = np.asarray(action_stats["q01"], dtype=np.float32)
        self.action_high = np.asarray(action_stats["q99"], dtype=np.float32)
        self.action_min = np.asarray(action_stats["min"], dtype=np.float32)
        self.action_max = np.asarray(action_stats["max"], dtype=np.float32)
        self.action_zero_mask = self.action_min == self.action_max

        self.records: List[Dict[str, Any]] = []
        with open(windows_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        if not self.records:
            raise ValueError(f"No windows found in {windows_path}")

        cache_window_size = int(self.metadata.get("window_size", self.window_size))
        if cache_window_size != self.window_size:
            raise ValueError(
                f"Cache window_size={cache_window_size} but finetune window_size={self.window_size}. "
                "Rebuild the cache or pass the matching --window_size."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rank, world_size = self._distributed_rank_world()
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        shard_id = rank * num_workers + worker_id
        num_shards = max(world_size * num_workers, 1)

        indices = list(range(len(self.records)))
        rng = random.Random(os.getpid() + shard_id)
        while True:
            if self.train:
                rng.shuffle(indices)
            for idx in indices:
                if idx % num_shards == shard_id:
                    yield self._make_instance(self.records[idx], rng)
            if not self.train:
                break

    @staticmethod
    def _distributed_rank_world() -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _make_instance(self, record: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        overlap = rng.randint(0, 1)
        frame_paths = record["frame_paths"]
        actions_raw = np.asarray(record["actions"], dtype=np.float32)
        if len(frame_paths) < self.window_size + 1:
            raise ValueError(f"Window has {len(frame_paths)} frames; expected at least {self.window_size + 1}.")
        if actions_raw.shape[0] < self.window_size + 1:
            raise ValueError(f"Window has {actions_raw.shape[0]} actions; expected at least {self.window_size + 1}.")

        input_idx = overlap
        target_idx = self.window_size - 1 + overlap
        hist_initial_idx = 0
        hist_target_idx = self.window_size - 1
        requested = {input_idx, target_idx, hist_initial_idx, hist_target_idx}
        images = {idx: self._load_image(frame_paths[idx], rng) for idx in requested}

        input_img = images[input_idx]
        target_img = images[target_idx]
        hist_initial_img = images[hist_initial_idx]
        hist_target_img = images[hist_target_idx]
        actions = self._normalize_actions(actions_raw[overlap : self.window_size + overlap])

        return {
            "pixel_values": self.image_transform(input_img),
            "lam_initial_pixel_values": self.image_transform_lam(input_img),
            "lam_target_pixel_values": self.image_transform_lam(target_img),
            "hist_lam_initial_pixel_values": self.image_transform_lam(hist_initial_img),
            "hist_lam_target_pixel_values": self.image_transform_lam(hist_target_img),
            "has_history": np.asarray(overlap > 0, dtype=np.bool_),
            "lang": str(record.get("task", self.metadata.get("task", ""))).lower(),
            "actions": actions.astype(np.float32, copy=False),
            "dataset_name": self.dataset_name,
        }

    def _load_image(self, rel_path: str, rng: random.Random) -> Image.Image:
        path = self.cache_dir / rel_path
        image = Image.open(path).convert("RGB")
        if self.image_aug:
            image = self._augment_image(image, rng)
        return image

    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        denom = self.action_high - self.action_low + 1e-8
        normalized = 2.0 * (actions - self.action_low) / denom - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)
        if np.any(self.action_zero_mask):
            normalized[:, self.action_zero_mask] = 0.0
        return normalized

    @staticmethod
    def _augment_image(image: Image.Image, rng: random.Random) -> Image.Image:
        # Mirrors the RLDS Bridge image augmentations: fixed 0.9 area crop, then color jitter.
        from torchvision.transforms import functional as F

        width, height = image.size
        crop = int(round(math.sqrt(0.9) * min(width, height)))
        crop = max(1, min(crop, width, height))
        top = rng.randint(0, max(height - crop, 0))
        left = rng.randint(0, max(width - crop, 0))
        image = F.resized_crop(image, top, left, crop, crop, (height, width), interpolation=_BICUBIC)
        image = F.adjust_brightness(image, rng.uniform(0.8, 1.2))
        image = F.adjust_contrast(image, rng.uniform(0.8, 1.2))
        image = F.adjust_saturation(image, rng.uniform(0.8, 1.2))
        image = F.adjust_hue(image, rng.uniform(-0.05, 0.05))
        return image
