import numpy as np
import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.robots.fr3.feature_adapter import fr3_observation_dataset_features
from lerobot.robots.fr3.processor_fr3 import (
    FR3PolicyObservationProcessorStep,
    prepare_fr3_array_for_policy,
)
from lerobot.utils.feature_utils import build_dataset_frame


def test_xense_online_and_offline_layouts_stay_independent():
    step = FR3PolicyObservationProcessorStep()
    observation = {
        "xense.sensor0.force_field": np.zeros((35, 20, 3), dtype=np.float32),
        "xense.sensor1.force_field": np.ones((2, 4, 35, 20, 3), dtype=np.float32),
    }
    processed = step.observation(observation)
    assert processed["xense.sensor0.force_field"].shape == (1, 3, 35, 20)
    assert processed["xense.sensor1.force_field"].shape == (2, 4, 3, 35, 20)

    features = {
        PipelineFeatureType.OBSERVATION: {
            "observation.xense.sensor0.force_field": PolicyFeature(type=FeatureType.STATE, shape=(35, 20, 3)),
            "observation.images.camera.cam1.depth": PolicyFeature(
                type=FeatureType.VISUAL, shape=(480, 640, 1)
            ),
        }
    }
    transformed = step.transform_features(features)[PipelineFeatureType.OBSERVATION]
    assert transformed["observation.xense.sensor0.force_field"].shape == (3, 35, 20)
    assert transformed["observation.images.camera.cam1.depth"].shape == (1, 480, 640)


def test_rgb_normalizes_and_depth_preserves_z16_scale():
    rgb = prepare_fr3_array_for_policy("camera.cam5.rgb", np.full((480, 640, 3), 255, dtype=np.uint8))
    depth = prepare_fr3_array_for_policy(
        "camera.cam5.depth", torch.full((480, 640, 1), 1200, dtype=torch.uint16)
    )
    assert rgb.shape == (1, 3, 480, 640)
    assert rgb.dtype == np.float32 and rgb.max() == 1.0
    assert depth.shape == (1, 1, 480, 640)
    assert depth.dtype == torch.float32 and depth.max() == 1200


def test_build_dataset_frame_preserves_numeric_array3d():
    features = fr3_observation_dataset_features(camera_count=5, use_videos=False)
    O_T_EE = np.array(
        [
            [0.0, -1.0, 0.0, 0.12],
            [1.0, 0.0, 0.0, -0.34],
            [0.0, 0.0, 1.0, 0.56],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    values = {
        **{f"fr3_joint{i}.pos": float(i) for i in range(1, 8)},
        "gripper.pos": 0.5,
        "fr3.dq": np.zeros(7, dtype=np.float32),
        "fr3.tau_J": np.zeros(7, dtype=np.float32),
        "fr3.O_T_EE": O_T_EE,
        "gripper.gPO": np.uint8(128),
        "gripper.gCU": np.uint8(0),
        "ft300s.wrench": np.zeros(6, dtype=np.float32),
        "xense.sensor0.force_field": np.zeros((35, 20, 3), dtype=np.float32),
        "xense.sensor1.force_field": np.ones((35, 20, 3), dtype=np.float32),
    }
    for i in range(1, 6):
        values[f"camera.cam{i}.rgb"] = np.zeros((480, 640, 3), dtype=np.uint8)
        values[f"camera.cam{i}.depth"] = np.zeros((480, 640, 1), dtype=np.uint16)
    frame = build_dataset_frame(features, values, prefix="observation")
    assert frame["observation.xense.sensor0.force_field"].shape == (35, 20, 3)
    assert frame["observation.xense.sensor1.force_field"].dtype == np.float32
    assert frame["observation.fr3.O_T_EE"].dtype == np.float32
    np.testing.assert_array_equal(frame["observation.fr3.O_T_EE"], O_T_EE)
