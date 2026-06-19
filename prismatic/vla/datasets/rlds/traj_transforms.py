"""
traj_transforms.py

Contains trajectory transforms used in the orca data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory length).
"""

import logging
import os
from typing import Dict

import tensorflow as tf

def chunk_act_obs(traj, window_size, future_action_window_size, lam_random_horizon=False, lam_window_size=10):
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]

    first_indices = tf.range(traj_len)[:, None]
    if lam_random_horizon:
        force_h2_offset = os.environ.get("POLAR_LAM_FORCE_H2_OFFSET")
        if force_h2_offset:
            force_h2_offset = int(force_h2_offset)
            if force_h2_offset < 0 or force_h2_offset >= lam_window_size:
                raise ValueError(
                    "POLAR_LAM_FORCE_H2_OFFSET must satisfy "
                    f"0 <= value < lam_window_size ({lam_window_size}), got {force_h2_offset}."
                )
            future_offsets = tf.fill([traj_len, 1], tf.cast(force_h2_offset, tf.int32))
            mid_offsets = tf.fill(
                [traj_len, 1],
                tf.cast(max(0, min(force_h2_offset, 1)), tf.int32),
            )
        else:
            future_offsets = tf.random.uniform([traj_len, 1], minval=2, maxval=lam_window_size, dtype=tf.int32)
            mid_offsets = tf.cast(
                tf.floor(tf.random.uniform([traj_len, 1]) * tf.cast(future_offsets - 1, tf.float32)),
                tf.int32,
            ) + 1
        chunk_indices = tf.concat(
            [first_indices, first_indices + mid_offsets, first_indices + future_offsets],
            axis=1,
        )
    else:
        last_indices = tf.maximum(first_indices + (window_size - 1), 0)
        chunk_indices = tf.concat([first_indices, last_indices], axis=1)

    # Create action_chunk_indices for the first and last elements
    action_first_indices = first_indices
    action_last_indices = tf.minimum(first_indices + (window_size + future_action_window_size - 1), traj_len - 1)
    action_chunk_indices = tf.concat([action_first_indices, action_last_indices], axis=1)  # Shape: [traj_len, 2]

    floored_chunk_indices = tf.maximum(tf.minimum(chunk_indices, traj_len - 1), 0)

    if "timestep" in traj["task"]:
        goal_timestep = traj["task"]["timestep"]
    else:
        goal_timestep = tf.fill([traj_len], traj_len - 1)

    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])
    if lam_random_horizon:
        future_action_indices = first_indices + future_offsets
        floored_future_action_indices = tf.minimum(
            tf.maximum(future_action_indices, 0),
            goal_timestep[:, None],
        )
        radprog_future_action = tf.gather(traj["action"], floored_future_action_indices)[:, 0]
        sequence_offsets = tf.range(lam_window_size, dtype=tf.int32)[None, :]
        sequence_indices = first_indices + sequence_offsets
        sequence_mask = sequence_offsets <= future_offsets
        floored_sequence_indices = tf.minimum(
            tf.maximum(sequence_indices, 0),
            goal_timestep[:, None],
        )
        radprog_action_sequence = tf.gather(traj["action"], floored_sequence_indices)

    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj["observation"])
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    traj["observation"]["pad_mask"] = chunk_indices >= 0
    if lam_random_horizon:
        traj["task"]["radprog_mid_offsets"] = tf.squeeze(mid_offsets, axis=1)
        traj["task"]["radprog_future_offsets"] = tf.squeeze(future_offsets, axis=1)
        traj["task"]["radprog_valid"] = tf.squeeze(first_indices + future_offsets < traj_len, axis=1)

    # If no absolute_action_mask was provided, assume all actions are relative
    if "absolute_action_mask" not in traj and future_action_window_size > 0:
        logging.warning(
            "future_action_window_size > 0 but no absolute_action_mask was provided. "
            "Assuming all actions are relative for the purpose of making neutral actions."
        )
    absolute_action_mask = traj.get("absolute_action_mask", tf.zeros([traj_len, action_dim], dtype=tf.bool))
    neutral_actions = tf.where(
        absolute_action_mask[:, None, :],
        traj["action"],  # absolute actions are repeated (already done during chunking)
        tf.zeros_like(traj["action"]),  # relative actions are zeroed
    )

    # Actions past the goal timestep become neutral
    action_past_goal = action_chunk_indices > goal_timestep[:, None]
    traj["action"] = tf.where(action_past_goal[:, :, None], neutral_actions, traj["action"])
    if lam_random_horizon:
        future_action_past_goal = future_action_indices > goal_timestep[:, None]
        neutral_future_action = tf.where(
            absolute_action_mask,
            radprog_future_action,
            tf.zeros_like(radprog_future_action),
        )
        traj["task"]["radprog_future_action"] = tf.where(
            future_action_past_goal,
            neutral_future_action,
            radprog_future_action,
        )
        sequence_past_goal = sequence_indices > goal_timestep[:, None]
        neutral_action_sequence = tf.where(
            absolute_action_mask[:, None, :],
            radprog_action_sequence,
            tf.zeros_like(radprog_action_sequence),
        )
        radprog_action_sequence = tf.where(
            sequence_past_goal[:, :, None],
            neutral_action_sequence,
            radprog_action_sequence,
        )
        traj["task"]["radprog_action_sequence"] = tf.where(
            sequence_mask[:, :, None],
            radprog_action_sequence,
            tf.zeros_like(radprog_action_sequence),
        )
        traj["task"]["radprog_action_sequence_mask"] = sequence_mask

    return traj


def chunk_act_obs_random_horizon(traj, window_size, future_action_window_size):
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]
    max_offset = tf.maximum(tf.cast(window_size - 1, tf.int32), 2)

    first_indices = tf.range(traj_len, dtype=tf.int32)
    remaining = tf.maximum(traj_len - 1 - first_indices, 0)
    max_valid_offset = tf.minimum(max_offset, remaining)
    valid = max_valid_offset >= 2

    h2_span = tf.maximum(max_valid_offset - 1, 1)
    sampled_h2 = tf.cast(
        tf.floor(tf.random.uniform(shape=[traj_len]) * tf.cast(h2_span, tf.float32)),
        tf.int32,
    ) + 2
    h2_offsets = tf.where(valid, sampled_h2, remaining)

    h1_span = tf.maximum(h2_offsets - 1, 1)
    sampled_h1 = tf.cast(
        tf.floor(tf.random.uniform(shape=[traj_len]) * tf.cast(h1_span, tf.float32)),
        tf.int32,
    ) + 1
    h1_offsets = tf.where(valid, sampled_h1, tf.zeros_like(sampled_h1))

    chunk_indices = tf.stack(
        [
            first_indices,
            first_indices + h1_offsets,
            first_indices + h2_offsets,
        ],
        axis=1,
    )
    action_chunk_indices = tf.stack(
        [
            first_indices,
            first_indices + h2_offsets + future_action_window_size,
        ],
        axis=1,
    )

    floored_chunk_indices = tf.maximum(tf.minimum(chunk_indices, traj_len - 1), 0)

    if "timestep" in traj["task"]:
        goal_timestep = traj["task"]["timestep"]
    else:
        goal_timestep = tf.fill([traj_len], traj_len - 1)

    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])

    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj["observation"])
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)
    traj["observation"]["pad_mask"] = chunk_indices >= 0
    traj["observation"]["lam_h1_offset"] = h1_offsets
    traj["observation"]["lam_h2_offset"] = h2_offsets
    traj["observation"]["lam_radprog_valid"] = valid

    if "absolute_action_mask" not in traj and future_action_window_size > 0:
        logging.warning(
            "future_action_window_size > 0 but no absolute_action_mask was provided. "
            "Assuming all actions are relative for the purpose of making neutral actions."
        )
    absolute_action_mask = traj.get("absolute_action_mask", tf.zeros([traj_len, action_dim], dtype=tf.bool))
    neutral_actions = tf.where(
        absolute_action_mask[:, None, :],
        traj["action"],
        tf.zeros_like(traj["action"]),
    )

    action_past_goal = action_chunk_indices > goal_timestep[:, None]
    traj["action"] = tf.where(action_past_goal[:, :, None], neutral_actions, traj["action"])

    return traj


def chunk_act_obs_window(
    traj: Dict,
    window_size: int,
    future_action_window_size: int = 0,
    lam_random_horizon: bool = False,
    lam_window_size: int = 10,
) -> Dict:
    """
    Chunks actions and observations into the given window_size.

    "observation" keys are given a new axis (at index 1) of size `window_size` containing `window_size - 1`
    observations from the past and the current observation. "action" is given a new axis (at index 1) of size
    `window_size + future_action_window_size` containing `window_size - 1` actions from the past, the current
    action, and `future_action_window_size` actions from the future. "pad_mask" is added to "observation" and
    indicates whether an observation should be considered padding (i.e. if it had come from a timestep
    before the start of the trajectory).
    """
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]
    chunk_indices = tf.broadcast_to(tf.range(-window_size + 1, 1), [traj_len, window_size]) + tf.broadcast_to(
        tf.range(traj_len)[:, None], [traj_len, window_size]
    )
    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(traj_len)[:, None],
        [traj_len, window_size + future_action_window_size],
    )

    floored_chunk_indices = tf.maximum(chunk_indices, 0)

    if "timestep" in traj["task"]:
        goal_timestep = traj["task"]["timestep"]
    else:
        goal_timestep = tf.fill([traj_len], traj_len - 1)

    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])

    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj["observation"])
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    # indicates whether an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # if no absolute_action_mask was provided, assume all actions are relative
    if "absolute_action_mask" not in traj and future_action_window_size > 0:
        logging.warning(
            "future_action_window_size > 0 but no absolute_action_mask was provided. "
            "Assuming all actions are relative for the purpose of making neutral actions."
        )
    absolute_action_mask = traj.get("absolute_action_mask", tf.zeros([traj_len, action_dim], dtype=tf.bool))
    neutral_actions = tf.where(
        absolute_action_mask[:, None, :],
        traj["action"],  # absolute actions are repeated (already done during chunking)
        tf.zeros_like(traj["action"]),  # relative actions are zeroed
    )

    # actions past the goal timestep become neutral
    action_past_goal = action_chunk_indices > goal_timestep[:, None]
    traj["action"] = tf.where(action_past_goal[:, :, None], neutral_actions, traj["action"])

    return traj


def subsample(traj: Dict, subsample_length: int) -> Dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)

    return traj


def add_pad_mask_dict(traj: Dict) -> Dict:
    """
    Adds a dictionary indicating which elements of the observation/task should be treated as padding.
        =>> traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]

    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            # Handles "language_instruction", "image_*", and "depth_*"
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0

            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]["pad_mask_dict"] = pad_mask_dict

    return traj
