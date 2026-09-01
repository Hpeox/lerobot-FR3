# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

JOINT_POSITION_KEYS = tuple(f"fr3_joint{i}.pos" for i in range(1, 8))
ACTION_KEYS = (*JOINT_POSITION_KEYS, "gripper.pos")
XENSE_KEYS = ("xense.sensor0.force_field", "xense.sensor1.force_field")


def fr3_camera_feature_keys(camera_count: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if isinstance(camera_count, bool) or not isinstance(camera_count, int):
        raise TypeError("camera_count must be an integer, not bool")
    if camera_count <= 0:
        raise ValueError("camera_count must be positive")
    rgb_keys = tuple(f"camera.cam{i}.rgb" for i in range(1, camera_count + 1))
    depth_keys = tuple(f"camera.cam{i}.depth" for i in range(1, camera_count + 1))
    return rgb_keys, depth_keys


def fr3_observation_dataset_features(
    *, camera_count: int, use_videos: bool = True
) -> dict[str, dict[str, Any]]:
    """Explicit FR3 schema that keeps Xense Array3D values out of the camera path."""

    rgb_keys, depth_keys = fr3_camera_feature_keys(camera_count)
    image_dtype = "video" if use_videos else "image"
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": list(ACTION_KEYS),
        },
        "observation.fr3.dq": {"dtype": "float32", "shape": (7,), "names": None},
        "observation.fr3.tau_J": {"dtype": "float32", "shape": (7,), "names": None},
        "observation.fr3.O_T_EE": {"dtype": "float32", "shape": (4, 4), "names": None},
        "observation.gripper.gPO": {"dtype": "uint8", "shape": (1,), "names": None},
        "observation.gripper.gCU": {"dtype": "uint8", "shape": (1,), "names": None},
        "observation.ft300s.wrench": {"dtype": "float32", "shape": (6,), "names": None},
        "observation.xense.sensor0.force_field": {
            "dtype": "float32",
            "shape": (35, 20, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.xense.sensor1.force_field": {
            "dtype": "float32",
            "shape": (35, 20, 3),
            "names": ["height", "width", "channels"],
        },
    }
    for key in rgb_keys:
        features[f"observation.images.{key}"] = {
            "dtype": image_dtype,
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        }
    for key in depth_keys:
        features[f"observation.images.{key}"] = {
            "dtype": image_dtype,
            "shape": (480, 640, 1),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True},
        }
    return features


def fr3_action_dataset_features() -> dict[str, dict[str, Any]]:
    return {
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": list(ACTION_KEYS),
        }
    }
