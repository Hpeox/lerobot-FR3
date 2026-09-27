"""Runtime-only adapter for a complete TactiGen-V5 checkpoint."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

from lerobot.policies.acmt_dp.gripper_mapping import fr3_pos_to_policy_gripper


def gello_opening_from_fr3_pos(fr3_pos: Tensor) -> Tensor:
    """Training GELLO opening is opposite to the policy's closedness."""

    return 1.0 - fr3_pos_to_policy_gripper(fr3_pos)


class TactiGenV5Runtime:
    """Keep V5 weights outside the ACMT-ACT policy state dict."""

    def __init__(self, checkpoint: str, bootstrap: str, device: str, expected_sha256: str | None = None):
        from tactigen.physics_contact.model import (
            LEGACY_WO_CORRECTION_LAYOUT,
            MAIN_LAYOUT,
            TactiGenV5Model,
        )

        checkpoint_path = Path(checkpoint)
        bootstrap_path = Path(bootstrap)
        if not checkpoint_path.is_file() or not bootstrap_path.is_file():
            raise FileNotFoundError("TactiGen-V5 checkpoint or DFormer bootstrap is missing")
        with checkpoint_path.open("rb") as stream:
            self.sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError("TactiGen-V5 checkpoint SHA-256 does not match deployment config")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("model_config"), Mapping):
            raise ValueError("TactiGen-V5 checkpoint is missing model_config")
        raw_config = dict(payload["model_config"])
        layout = raw_config.pop("layout", None)
        legacy = layout == LEGACY_WO_CORRECTION_LAYOUT
        if layout not in {MAIN_LAYOUT, LEGACY_WO_CORRECTION_LAYOUT}:
            raise ValueError(f"Unsupported TactiGen-V5 checkpoint layout: {layout!r}")
        if legacy and raw_config.get("ablation_variant") != "wo_learned_correction":
            raise ValueError("Only the gear wo_learned_correction ablation is supported")
        expected_protocol = {
            "t_obs": 4,
            "t_pred": 1,
            "contact_grid": [35, 20],
            "force_grid": [35, 20],
            "force_order": ["fx", "fy", "fz"],
            "uses_ft300_input": False,
        }
        if any(raw_config.get(key) != value for key, value in expected_protocol.items()):
            raise ValueError("TactiGen-V5 checkpoint input/output protocol is incompatible")
        if raw_config.get("uses_learned_correction") not in {False, None}:
            raise ValueError("TactiGen-V5 Learned correction is not supported")
        self.model_config = dict(payload["model_config"])
        for key in (
            "t_obs", "t_pred", "contact_grid", "force_grid", "force_order",
            "uses_ft300_input", "dformer_trainable_stage", "view_dropout",
            "uses_learned_correction", "ablation_variant", "ablated_component",
            "final_constraint_projection", "direct_force_head_normal_activation",
            "dformerv2_repo_path", "dformerv2_checkpoint",
        ):
            raw_config.pop(key, None)
        self.model = TactiGenV5Model(
            **raw_config, dformerv2_checkpoint=str(bootstrap_path)
        )
        state = payload.get("model_state_dict")
        if not isinstance(state, Mapping):
            raise ValueError("TactiGen-V5 checkpoint is missing model_state_dict")
        state = dict(state)
        if legacy:
            state = {
                key: value for key, value in state.items()
                if not key.startswith("force_residual_head.") and key != "force_residual_scale"
            }
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if torch.device(device).type == "cuda":
            # First CUDA execution initializes kernels and caches, which can
            # take longer than the first 8-action reserve. Pay that cost
            # while loading the artifact, before any robot command is sent.
            self._warmup(device)

    def _warmup(self, device: str) -> None:
        identity = torch.eye(4, device=device).reshape(1, 1, 4, 4).expand(1, 4, 4, 4)
        self.predict_next(
            {
                "rgb": torch.zeros(1, 4, 2, 3, 480, 640, device=device),
                "depth": torch.zeros(1, 4, 2, 1, 480, 640, device=device),
                "lowdim": torch.zeros(1, 4, 15, device=device),
            },
            identity,
            torch.zeros(1, 8, device=device),
            torch.zeros(1, 4, 8, device=device),
        )
        torch.cuda.synchronize(device)

    def reset(self) -> None:
        pass

    @torch.inference_mode()
    def predict_next(
        self,
        observation: dict[str, Tensor],
        pose: Tensor,
        action: Tensor,
        command_history: Tensor,
    ) -> Tensor:
        from tactigen.force_field_dataset import WRIST_ROI, _matrix_to_quaternion_xyzw

        rgb = observation["rgb"]
        depth = observation["depth"]
        lowdim = observation["lowdim"]
        if tuple(rgb.shape[1:4]) != (4, 2, 3) or tuple(rgb.shape[-2:]) != (480, 640):
            raise ValueError("V5 wrist RGB history must be [B,4,2,3,480,640]")
        if tuple(depth.shape[1:4]) != (4, 2, 1) or tuple(depth.shape[-2:]) != (480, 640):
            raise ValueError("V5 wrist depth history must be [B,4,2,1,480,640]")
        if tuple(lowdim.shape[1:]) != (4, 15) or tuple(pose.shape[1:]) != (4, 4, 4):
            raise ValueError("V5 robot history must contain four q/dq/gPO/pose samples")
        if tuple(command_history.shape[1:]) != (4, 8) or tuple(action.shape[1:]) != (8,):
            raise ValueError("V5 command history/action shape mismatch")
        y_roi, x_roi = WRIST_ROI
        batch_size = pose.shape[0]
        transforms = pose.detach().cpu().numpy().reshape(-1, 4, 4)
        quaternions = torch.as_tensor(
            _matrix_to_quaternion_xyzw(transforms),
            dtype=pose.dtype, device=pose.device,
        ).reshape(batch_size, 4, 4)
        ee_pose = torch.cat((pose[:, :, :3, 3], quaternions), dim=-1)
        batch = {
            "realsense.cam1_color": rgb[:, :, 0, :, y_roi, x_roi],
            "realsense.cam1_depth": depth[:, :, 0, :, y_roi, x_roi],
            "realsense.cam2_color": rgb[:, :, 1, :, y_roi, x_roi],
            "realsense.cam2_depth": depth[:, :, 1, :, y_roi, x_roi],
            "robot.q": lowdim[:, :, :7],
            "robot.dq": lowdim[:, :, 7:14],
            "robot.O_T_EE": ee_pose,
            "gripper.gripper_gPO": lowdim[:, :, 14:15],
            "gello.history_q": command_history[:, :, :7],
            "gello.history_gripper_width": command_history[:, :, 7:8],
            "gello.future_q": action[:, None, :7],
            "gello.future_gripper_width": action[:, None, 7:8],
        }
        force = self.model.inference(batch)["force"]
        if tuple(force.shape) != (batch_size, 1, 2, 35, 20, 3) or not torch.isfinite(force).all():
            raise ValueError("TactiGen-V5 produced an invalid force field")
        return force[:, 0]
