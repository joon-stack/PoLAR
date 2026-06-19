"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import os
import random
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        # print(labels)
        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name)


@dataclass
class RLDSBatchTransformWindowActionWithHistory:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    image_transform_lam: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    window_size: int = 5

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        # img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        randomized_overlap = random.randint(0,1)
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        img_k = Image.fromarray(rlds_batch["observation"]["image_primary"][self.window_size-1])

        input_img = Image.fromarray(rlds_batch["observation"]["image_primary"][randomized_overlap])
        pixel_values = self.image_transform(input_img)

        if (
            os.environ.get("POLAR_BATCH_LAM_IN_COLLATOR", "0") == "1"
            or os.environ.get("POLAR_LAM_IN_TRAIN_LOOP", "0") == "1"
        ):
            target_img = Image.fromarray(
                rlds_batch["observation"]["image_primary"][self.window_size - 1 + randomized_overlap]
            )
            return dict(
                pixel_values=pixel_values,
                lam_initial_pixel_values=self.image_transform_lam(input_img),
                lam_target_pixel_values=self.image_transform_lam(target_img),
                hist_lam_initial_pixel_values=self.image_transform_lam(img),
                hist_lam_target_pixel_values=self.image_transform_lam(img_k),
                has_history=np.array(randomized_overlap > 0, dtype=np.bool_),
                lang=lang,
                actions=rlds_batch["action"][randomized_overlap: self.window_size + randomized_overlap],
                dataset_name=dataset_name,
            )

        with torch.no_grad():
            initial_pixel_values = self.image_transform_lam(input_img)
            target_pixel_values= self.image_transform_lam(Image.fromarray(rlds_batch["observation"]["image_primary"][self.window_size - 1 + randomized_overlap]))

            video = torch.stack([initial_pixel_values, target_pixel_values], dim=0).unsqueeze(0).to(self.action_tokenizer.device)
            latent_action_idx = self.action_tokenizer.vq_encode(video)['indices'].squeeze()

            if randomized_overlap > 0:
                initial_pixel_values = self.image_transform_lam(img)
                target_pixel_values= self.image_transform_lam(img_k)
                video = torch.stack([initial_pixel_values, target_pixel_values], dim=0).unsqueeze(0).to(self.action_tokenizer.device)
                hist_action_idx = self.action_tokenizer.vq_encode(video)['indices'].squeeze()        

        action_vocab = [f'<ACT_{i.item()}>' for i in latent_action_idx]   # [ACT_1, ACT_2, ... ACT_K]
        # print(action_vocab)
        action_tokens = ''
        for i, action in enumerate(action_vocab):
            action_tokens += action

        input_prompt = f"What action should the robot take to {lang}?"
        if randomized_overlap > 0:
            action_vocab = [f'<ACT_{i.item()}>' for i in hist_action_idx]   # [ACT_1, ACT_2, ... ACT_K]

            hist_action_tokens = ''
            for i, action in enumerate(action_vocab):
                hist_action_tokens += action

            input_prompt = f"What action should the robot take to {lang}? History action " + hist_action_tokens

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": input_prompt},
            {"from": "gpt", "value": action_tokens},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)


        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action_vocab) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, actions=rlds_batch["action"][randomized_overlap: self.window_size + randomized_overlap], latent_action_idx=latent_action_idx, dataset_name=dataset_name)


@dataclass
class RLDSBatchTransformLatentActionInputs:
    image_transform: ImageTransform
    image_transform_lam: ImageTransform

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare image/language inputs; the collator batches LAM pseudo-labeling."""
        dataset_name = rlds_batch["dataset_name"]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        img_k = Image.fromarray(rlds_batch["observation"]["image_primary"][-1])

        return dict(
            pixel_values=self.image_transform(img),
            initial_lam_pixel_values=self.image_transform_lam(img),
            target_lam_pixel_values=self.image_transform_lam(img_k),
            lang=lang,
            dataset_name=dataset_name,
        )


@dataclass
class RLDSBatchTransformVideo:
    image_transform: ImageTransform

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], np.array(rlds_batch["action"])
        
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])#.copy()
        initial_pixel_values = self.image_transform(img)

        target_frame_index = -1
        img_k = Image.fromarray(rlds_batch["observation"]["image_primary"][target_frame_index])#.copy()
        target_pixel_values= self.image_transform(img_k)

        output = dict(initial_pixel_values=initial_pixel_values, target_pixel_values=target_pixel_values, 
                      task_instruction=lang, action=action, dataset_name=dataset_name)
        if "radprog_mid_offsets" in rlds_batch["task"]:
            mid_img = Image.fromarray(rlds_batch["observation"]["image_primary"][1])
            output.update(
                radprog_mid_pixel_values=self.image_transform(mid_img),
                radprog_mid_offsets=np.array(rlds_batch["task"]["radprog_mid_offsets"], dtype=np.int64),
                radprog_future_offsets=np.array(rlds_batch["task"]["radprog_future_offsets"], dtype=np.int64),
                radprog_valid=np.array(rlds_batch["task"]["radprog_valid"], dtype=np.float32),
                radprog_future_action=np.array(rlds_batch["task"]["radprog_future_action"], dtype=np.float32),
                radprog_action_sequence=np.array(rlds_batch["task"]["radprog_action_sequence"], dtype=np.float32),
                radprog_action_sequence_mask=np.array(
                    rlds_batch["task"]["radprog_action_sequence_mask"],
                    dtype=np.float32,
                ),
            )

        return output




class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        window_size: int = 10,
        train: bool = True,
        image_aug: bool = False,
        training_phase: str = 'lam',
        lam_random_horizon: bool = False,
        lam_window_size: int = 10,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        normalization_name = os.environ.get("POLAR_RLDS_NORMALIZATION", "BOUNDS_Q99").upper()
        normalization_type = getattr(NormalizationType, normalization_name)

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=normalization_type,
        )
        stats_path = os.environ.get("POLAR_RLDS_STATS_PATH")
        if stats_path:
            for dataset_kwargs in per_dataset_kwargs:
                dataset_kwargs["dataset_statistics"] = stats_path
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                            # If we wanted to feed / predict more than one step
                future_action_window_size=0,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
                lam_random_horizon=lam_random_horizon,
                lam_window_size=lam_window_size,
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=int(os.environ.get("POLAR_FRAME_THREADS", "8")),
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=int(os.environ.get("POLAR_TRAJ_THREADS", str(len(mixture_spec)))),
            traj_read_threads=int(os.environ.get("POLAR_TRAJ_READ_THREADS", str(len(mixture_spec)))),
            train=train,
            training_phase=training_phase,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        dataset = self.dataset
        worker_info = get_worker_info()
        if worker_info is not None:
            dataset = dataset.shard(worker_info.num_workers, worker_info.id)
        profile_enabled = int(os.environ.get("POLAR_LAM_PROFILE_STEPS", "0")) > 0
        iterator = dataset.as_numpy_iterator()
        while True:
            next_start = time.perf_counter() if profile_enabled else None
            try:
                rlds_batch = next(iterator)
            except StopIteration:
                return
            next_done = time.perf_counter() if profile_enabled else None

            transform_start = time.perf_counter() if profile_enabled else None
            out = self.batch_transform(rlds_batch)
            if profile_enabled:
                transform_done = time.perf_counter()
                out["_profile"] = {
                    "rlds_next_s": next_done - next_start,
                    "batch_transform_s": transform_done - transform_start,
                }
            yield out

    def __len__(self) -> int:
        length_override = os.environ.get("POLAR_RLDS_LEN_OVERRIDE")
        if length_override:
            return int(length_override)
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        dataset = self.dataset
        worker_info = get_worker_info()
        if worker_info is not None:
            dataset = dataset.shard(worker_info.num_workers, worker_info.id)
        for rlds_batch in dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
