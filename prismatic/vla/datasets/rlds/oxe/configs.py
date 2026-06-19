"""Dataset configs for the PoLAR Bridge/Simpler RLDS path."""

from enum import IntEnum


class StateEncoding(IntEnum):
    NONE = -1
    POS_EULER = 1
    POS_QUAT = 2
    JOINT = 3
    JOINT_BIMANUAL = 4


class ActionEncoding(IntEnum):
    EEF_POS = 1
    JOINT_POS = 2
    JOINT_POS_BIMANUAL = 3
    EEF_R6 = 4


_BRIDGE_CONFIG = {
    "image_obs_keys": {"primary": "image_0", "secondary": "image_1", "wrist": None},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["EEF_state", None, "gripper_state"],
    "state_encoding": StateEncoding.POS_EULER,
    "action_encoding": ActionEncoding.EEF_POS,
}

_SIMPLER_CONFIG = {
    "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": [None, None, None],
    "state_encoding": StateEncoding.NONE,
    "action_encoding": ActionEncoding.EEF_POS,
}


OXE_DATASET_CONFIGS = {
    "bridge_oxe": {
        "image_obs_keys": {"primary": "image", "secondary": "image_1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "bridge_orig": dict(_BRIDGE_CONFIG),
    "bridge_dataset": dict(_BRIDGE_CONFIG),
    "carrot": dict(_SIMPLER_CONFIG),
    "eggplant": dict(_SIMPLER_CONFIG),
    "spoon": dict(_SIMPLER_CONFIG),
    "stack": dict(_SIMPLER_CONFIG),
}
