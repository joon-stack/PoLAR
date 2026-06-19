"""Standardization transforms for the PoLAR Bridge/Simpler RLDS path."""

from typing import Any, Dict

import tensorflow as tf

from prismatic.vla.datasets.rlds.utils.data_utils import binarize_gripper_actions, relabel_bridge_actions


def bridge_oxe_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize the Bridge V2 variant stored in Open X-Embodiment."""
    for key in trajectory.keys():
        if key == "traj_metadata":
            continue
        if key in ["observation", "action"]:
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    trajectory = relabel_bridge_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def bridge_orig_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize the original Bridge V2 TFDS release."""
    for key in trajectory.keys():
        if key == "traj_metadata":
            continue
        if key == "observation":
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory = relabel_bridge_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def simpler_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["language_instruction"] = trajectory["observation"]["language_instruction"]
    return trajectory


OXE_STANDARDIZATION_TRANSFORMS = {
    "bridge_oxe": bridge_oxe_dataset_transform,
    "bridge_orig": bridge_orig_dataset_transform,
    "bridge_dataset": bridge_orig_dataset_transform,
    "carrot": simpler_dataset_transform,
    "eggplant": simpler_dataset_transform,
    "spoon": simpler_dataset_transform,
    "stack": simpler_dataset_transform,
}
