"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""
import re
import string
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def select_lam_token_indices(outputs: Dict[str, torch.Tensor], token_view: str, batch_size: int) -> torch.Tensor:
    token_view = (token_view or "indices").strip().lower()
    if token_view == "indices":
        key = "indices"
    elif token_view == "factorized_direction":
        key = "direction_indices"
    else:
        raise ValueError(f"Unsupported lam_token_view={token_view!r}.")
    if key not in outputs:
        raise ValueError(f"LAM output missing {key!r} for lam_token_view={token_view!r}.")
    return outputs[key].view(batch_size, -1)


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output


@dataclass
class PaddedCollatorForLatentActionPrediction:
    model_max_length: int
    pad_token_id: int
    tokenizer: Any
    latent_action_model: torch.nn.Module
    prompt_builder_fn: Callable
    padding_side: str = "right"
    predict_stop_token: bool = True
    latent_action_token_len: int = 0
    lam_token_view: str = "indices"

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        profile_steps = int(os.environ.get("POLAR_PROFILE_STEPS", "0"))
        profile_start = time.perf_counter() if profile_steps > 0 else None
        pixel_values = [instance["pixel_values"] for instance in instances]
        dataset_names = [instance["dataset_name"] for instance in instances]

        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(instances))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
        profile_pixel_done = time.perf_counter() if profile_steps > 0 else None

        if os.environ.get("POLAR_DUMMY_LATENT_ACTIONS", "0") == "1":
            token_len = self.latent_action_token_len if self.latent_action_token_len > 0 else 1
            latent_action_idx = torch.zeros((len(instances), token_len), dtype=torch.long)
        elif "initial_lam_pixel_values" in instances[0]:
            initial = torch.stack([instance["initial_lam_pixel_values"] for instance in instances])
            target = torch.stack([instance["target_lam_pixel_values"] for instance in instances])
            device = next(self.latent_action_model.parameters()).device
            video = torch.stack([initial, target], dim=1).to(device, non_blocking=True)
            with torch.inference_mode():
                outputs = self.latent_action_model.vq_encode(video)
                latent_action_idx = select_lam_token_indices(outputs, self.lam_token_view, len(instances))
        else:
            token_len = self.latent_action_token_len if self.latent_action_token_len > 0 else 1
            latent_action_idx = torch.zeros((len(instances), token_len), dtype=torch.long)
        if self.latent_action_token_len > 0 and latent_action_idx.shape[-1] != self.latent_action_token_len:
            raise ValueError(
                "Latent action token length mismatch: "
                f"expected {self.latent_action_token_len}, got {latent_action_idx.shape[-1]}."
            )
        profile_lam_done = time.perf_counter() if profile_steps > 0 else None

        input_ids, labels = [], []
        for instance, action_indices in zip(instances, latent_action_idx.cpu()):
            action_tokens = "".join(f"<ACT_{idx}>" for idx in action_indices.tolist())
            prompt_builder = self.prompt_builder_fn("openvla")
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {instance['lang']}?"},
                {"from": "gpt", "value": action_tokens},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            cur_input_ids = self.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
            cur_labels = list(cur_input_ids)
            cur_input_ids, cur_labels = torch.tensor(cur_input_ids), torch.tensor(cur_labels)
            cur_labels[: -(len(action_indices) + 1)] = IGNORE_INDEX
            if not self.predict_stop_token:
                cur_labels[-1] = IGNORE_INDEX
            input_ids.append(cur_input_ids)
            labels.append(cur_labels)
        profile_token_done = time.perf_counter() if profile_steps > 0 else None

        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]
        attention_mask = input_ids.ne(self.pad_token_id)

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            dataset_names=dataset_names,
            latent_action_token_len=self.latent_action_token_len,
        )
        if profile_steps > 0:
            profile_end = time.perf_counter()
            output["profile"] = {
                "collator_total": profile_end - profile_start,
                "collator_pixel_stack": profile_pixel_done - profile_start,
                "collator_lam": profile_lam_done - profile_pixel_done,
                "collator_tokenize": profile_token_done - profile_lam_done,
                "collator_pad": profile_end - profile_token_done,
            }
        return output


@dataclass
class PaddedCollatorForWindowActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32
    base_tokenizer: Any = None
    latent_action_model: Any = None
    prompt_builder_fn: Any = None
    predict_stop_token: bool = True
    latent_action_token_len: int = 0
    lam_token_view: str = "indices"

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if "lam_initial_pixel_values" in instances[0]:
            train_loop_lam = os.environ.get("POLAR_LAM_IN_TRAIN_LOOP", "0") == "1"
            if not train_loop_lam and (
                self.base_tokenizer is None or self.latent_action_model is None or self.prompt_builder_fn is None
            ):
                raise ValueError("Batched LAM collation requires base_tokenizer, latent_action_model, and prompt_builder_fn.")

            profile_steps = int(os.environ.get("POLAR_PROFILE_STEPS", "0"))
            profile_start = time.perf_counter() if profile_steps > 0 else None

            pixel_values = [instance["pixel_values"] for instance in instances]
            if isinstance(pixel_values[0], torch.Tensor):
                pixel_values = torch.stack(pixel_values)
            elif isinstance(pixel_values[0], dict):
                pixel_values = {
                    k: torch.stack([pixel_values[idx][k] for idx in range(len(instances))]) for k in pixel_values[0]
                }
            else:
                raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
            profile_pixel_done = time.perf_counter() if profile_steps > 0 else None

            actions = torch.stack([torch.from_numpy(instance["actions"]) for instance in instances], dim=0)

            if train_loop_lam:
                output = dict(
                    pixel_values=pixel_values,
                    lam_initial_pixel_values=torch.stack([instance["lam_initial_pixel_values"] for instance in instances]),
                    lam_target_pixel_values=torch.stack([instance["lam_target_pixel_values"] for instance in instances]),
                    hist_lam_initial_pixel_values=torch.stack(
                        [instance["hist_lam_initial_pixel_values"] for instance in instances]
                    ),
                    hist_lam_target_pixel_values=torch.stack(
                        [instance["hist_lam_target_pixel_values"] for instance in instances]
                    ),
                    has_history=torch.as_tensor(
                        [bool(np.asarray(instance["has_history"]).reshape(-1)[0]) for instance in instances],
                        dtype=torch.bool,
                    ),
                    langs=[instance["lang"] for instance in instances],
                    actions=actions,
                    dataset_names=[instance["dataset_name"] for instance in instances],
                )
                if profile_steps > 0:
                    profile_end = time.perf_counter()
                    output["profile"] = {
                        "collator_total": profile_end - profile_start,
                        "collator_pixel_stack": profile_pixel_done - profile_start,
                        "collator_lam": 0.0,
                        "collator_tokenize": 0.0,
                        "collator_pad": profile_end - profile_pixel_done,
                    }
                return output

            if self.base_tokenizer is None or self.latent_action_model is None or self.prompt_builder_fn is None:
                raise ValueError("Batched LAM collation requires base_tokenizer, latent_action_model, and prompt_builder_fn.")

            device = next(self.latent_action_model.parameters()).device
            initial = torch.stack([instance["lam_initial_pixel_values"] for instance in instances]).to(device, non_blocking=True)
            target = torch.stack([instance["lam_target_pixel_values"] for instance in instances]).to(device, non_blocking=True)
            video = torch.stack([initial, target], dim=1)
            with torch.inference_mode():
                outputs = self.latent_action_model.vq_encode(video)
                latent_action_idx = select_lam_token_indices(outputs, self.lam_token_view, len(instances)).cpu()

            has_history = [bool(np.asarray(instance["has_history"]).reshape(-1)[0]) for instance in instances]
            hist_action_idx = torch.zeros_like(latent_action_idx)
            hist_indices = [idx for idx, enabled in enumerate(has_history) if enabled]
            if hist_indices:
                hist_initial = torch.stack(
                    [instances[idx]["hist_lam_initial_pixel_values"] for idx in hist_indices]
                ).to(device, non_blocking=True)
                hist_target = torch.stack(
                    [instances[idx]["hist_lam_target_pixel_values"] for idx in hist_indices]
                ).to(device, non_blocking=True)
                hist_video = torch.stack([hist_initial, hist_target], dim=1)
                with torch.inference_mode():
                    outputs = self.latent_action_model.vq_encode(hist_video)
                    encoded_hist = select_lam_token_indices(outputs, self.lam_token_view, len(hist_indices)).cpu()
                hist_action_idx[hist_indices] = encoded_hist
            profile_lam_done = time.perf_counter() if profile_steps > 0 else None

            input_ids, labels = [], []
            for instance, action_indices, hist_indices_for_sample, use_history in zip(
                instances, latent_action_idx, hist_action_idx, has_history
            ):
                if self.latent_action_token_len > 0 and action_indices.numel() != self.latent_action_token_len:
                    raise ValueError(
                        "Latent action token length mismatch: "
                        f"expected {self.latent_action_token_len}, got {action_indices.numel()}."
                    )
                action_tokens = "".join(f"<ACT_{idx}>" for idx in action_indices.tolist())
                input_prompt = f"What action should the robot take to {instance['lang']}?"
                if use_history:
                    hist_action_tokens = "".join(f"<ACT_{idx}>" for idx in hist_indices_for_sample.tolist())
                    input_prompt += " History action " + hist_action_tokens

                prompt_builder = self.prompt_builder_fn("openvla")
                conversation = [
                    {"from": "human", "value": input_prompt},
                    {"from": "gpt", "value": action_tokens},
                ]
                for turn in conversation:
                    prompt_builder.add_turn(turn["from"], turn["value"])

                cur_input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
                cur_labels = list(cur_input_ids)
                cur_input_ids, cur_labels = torch.tensor(cur_input_ids), torch.tensor(cur_labels)
                cur_labels[: -(len(action_indices) + 1)] = IGNORE_INDEX
                if not self.predict_stop_token:
                    cur_labels[-1] = IGNORE_INDEX
                input_ids.append(cur_input_ids)
                labels.append(cur_labels)
            profile_token_done = time.perf_counter() if profile_steps > 0 else None

            assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]
            attention_mask = input_ids.ne(self.pad_token_id)

            output = dict(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                actions=actions,
                latent_action_idx=latent_action_idx,
            )
            if "dataset_name" in instances[0]:
                output["dataset_names"] = [instance["dataset_name"] for instance in instances]
            if profile_steps > 0:
                profile_end = time.perf_counter()
                output["profile"] = {
                    "collator_total": profile_end - profile_start,
                    "collator_pixel_stack": profile_pixel_done - profile_start,
                    "collator_lam": profile_lam_done - profile_pixel_done,
                    "collator_tokenize": profile_token_done - profile_lam_done,
                    "collator_pad": profile_end - profile_token_done,
                }
            return output

        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # For low-level policy training
        actions = [torch.from_numpy(instance["actions"]) for instance in instances]
        actions = torch.stack(actions, dim=0)

        # Get latent action indexes
        latent_action_idx = [instance["latent_action_idx"] for instance in instances]
        latent_action_idx = torch.stack(latent_action_idx, dim=0)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            latent_action_idx=latent_action_idx,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output


@dataclass
class CollatorForLatentAction:
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        profile_enabled = int(os.environ.get("POLAR_LAM_PROFILE_STEPS", "0")) > 0
        profile_start = time.perf_counter() if profile_enabled else None
        sample_profiles = [instance.get("_profile") for instance in instances if "_profile" in instance]

        def scalar_value(value: Any) -> Any:
            return np.asarray(value).reshape(-1)[0].item()
        
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        initial_pixel_values = [instance["initial_pixel_values"] for instance in instances]
        initial_pixel_values = torch.stack(initial_pixel_values)
        
        target_pixel_values = [instance["target_pixel_values"] for instance in instances]
        target_pixel_values = torch.stack(target_pixel_values)
        pixel_values = torch.stack([initial_pixel_values, target_pixel_values], dim=1)
        profile_videos_done = time.perf_counter() if profile_enabled else None

        action = [torch.from_numpy(instance["action"]) for instance in instances]
        action = torch.stack(action)
        profile_action_done = time.perf_counter() if profile_enabled else None

        # removing all punctuation in task instruction
        task_instruction = [re.sub('[{}]'.format(string.punctuation),"",instance["task_instruction"]) for instance in instances]
        profile_text_done = time.perf_counter() if profile_enabled else None

        output = dict(
            videos=pixel_values,
            task_instruction=task_instruction,
            action=action,
        )
        if "radprog_mid_pixel_values" in instances[0]:
            output.update(
                radprog_mid_pixel_values=torch.stack([instance["radprog_mid_pixel_values"] for instance in instances]),
                radprog_mid_offsets=torch.as_tensor(
                    [scalar_value(instance["radprog_mid_offsets"]) for instance in instances],
                    dtype=torch.long,
                ),
                radprog_future_offsets=torch.as_tensor(
                    [scalar_value(instance["radprog_future_offsets"]) for instance in instances],
                    dtype=torch.long,
                ),
                radprog_valid=torch.as_tensor(
                    [scalar_value(instance["radprog_valid"]) for instance in instances],
                    dtype=torch.float32,
                ),
                radprog_future_action=torch.stack(
                    [torch.from_numpy(instance["radprog_future_action"]) for instance in instances]
                ),
                radprog_action_sequence=torch.stack(
                    [torch.from_numpy(instance["radprog_action_sequence"]) for instance in instances]
                ),
                radprog_action_sequence_mask=torch.stack(
                    [torch.from_numpy(instance["radprog_action_sequence_mask"]) for instance in instances]
                ),
            )
        profile_radprog_done = time.perf_counter() if profile_enabled else None
        if dataset_names is not None:
            output["dataset_names"] = dataset_names

        if profile_enabled:
            profile_end = time.perf_counter()
            rlds_next = [float(profile.get("rlds_next_s", 0.0)) for profile in sample_profiles]
            batch_transform = [float(profile.get("batch_transform_s", 0.0)) for profile in sample_profiles]
            output["_profile"] = {
                "batch_size": torch.tensor(float(len(instances))),
                "rlds_next_sum": torch.tensor(float(sum(rlds_next))),
                "rlds_next_item_max": torch.tensor(float(max(rlds_next, default=0.0))),
                "batch_transform_sum": torch.tensor(float(sum(batch_transform))),
                "batch_transform_item_max": torch.tensor(float(max(batch_transform, default=0.0))),
                "collate_total": torch.tensor(float(profile_end - profile_start)),
                "collate_videos": torch.tensor(float(profile_videos_done - profile_start)),
                "collate_action": torch.tensor(float(profile_action_done - profile_videos_done)),
                "collate_text": torch.tensor(float(profile_text_done - profile_action_done)),
                "collate_radprog": torch.tensor(float(profile_radprog_done - profile_text_done)),
            }

        return output


@dataclass
class CollatorForMultiViewVideo:
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        initial_pixel_values = [instance["initial_pixel_values"] for instance in instances]
        initial_pixel_values = torch.stack(initial_pixel_values)
        
        target_pixel_values = [instance["target_pixel_values"] for instance in instances]
        target_pixel_values = torch.stack(target_pixel_values)
        pixel_values = torch.stack([initial_pixel_values, target_pixel_values], dim=1)


        initial_pixel_values_view2 = [instance["initial_pixel_values_view2"] for instance in instances]
        initial_pixel_values_view2 = torch.stack(initial_pixel_values_view2)
        
        target_pixel_values_view2 = [instance["target_pixel_values_view2"] for instance in instances]
        target_pixel_values_view2 = torch.stack(target_pixel_values_view2)
        pixel_values_view2 = torch.stack([initial_pixel_values_view2, target_pixel_values_view2], dim=1)
        


        action = [torch.from_numpy(instance["action"]) for instance in instances]
        action = torch.stack(action)

        # removing all punctuation in task instruction
        task_instruction = [re.sub('[{}]'.format(string.punctuation),"",instance["task_instruction"]) for instance in instances]


        output = dict(
            videos=pixel_values,
            videos_view2=pixel_values_view2,
            task_instruction=task_instruction,
            action=action,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names

        return output
