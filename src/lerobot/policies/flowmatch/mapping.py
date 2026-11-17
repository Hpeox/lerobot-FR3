"""Explicit GELLO-label -> named FR3 target-space conversion, no hardware I/O."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActionMapping:
    joint_order: tuple
    scale: tuple
    offset_rad: tuple
    gpo_at_raw_zero: float
    gpo_at_raw_one: float
    verification_source: str

    def __post_init__(self):
        if (
            len(self.joint_order) != 7
            or set(self.joint_order) != set(range(7))
            or any(type(i) is not int for i in self.joint_order)
        ):
            raise ValueError("joint_order must be a permutation of 0..6")
        scale, offset = torch.as_tensor(self.scale), torch.as_tensor(self.offset_rad)
        if (
            scale.shape != (7,)
            or offset.shape != (7,)
            or not torch.isfinite(scale).all()
            or not torch.isfinite(offset).all()
        ):
            raise ValueError("Expected seven finite joint scales/offsets")
        if (scale == 0).any():
            raise ValueError("Joint scales cannot be zero")
        endpoints = torch.as_tensor([self.gpo_at_raw_zero, self.gpo_at_raw_one])
        if (
            not torch.isfinite(endpoints).all()
            or (endpoints < 0).any()
            or (endpoints > 255).any()
            or endpoints[0] == endpoints[1]
        ):
            raise ValueError("Provide two distinct measured gPO endpoints in [0,255]")
        if not isinstance(self.verification_source, str) or not self.verification_source.strip():
            raise ValueError("Record where the actual action mapping was verified")

    @classmethod
    def from_dict(cls, value):
        try:
            return cls(
                tuple(value["joint_order"]),
                tuple(value["scale"]),
                tuple(value["offset_rad"]),
                value["gpo_at_raw_zero"],
                value["gpo_at_raw_one"],
                value["verification_source"],
            )
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Action mapping is unconfigured; raw GELLO targets are not FR3 commands"
            ) from error

    def convert(self, action: torch.Tensor):
        if action.shape[-1] != 8 or not torch.isfinite(action).all():
            raise ValueError("Expected finite raw actions with last dimension 8")
        result = action.clone()
        scale = torch.as_tensor(self.scale, dtype=action.dtype, device=action.device)
        offset = torch.as_tensor(self.offset_rad, dtype=action.dtype, device=action.device)
        result[..., :7] = action[..., list(self.joint_order)] * scale + offset
        grip = action[..., 7].clamp(0, 1)
        result[..., 7] = (self.gpo_at_raw_zero + grip * (self.gpo_at_raw_one - self.gpo_at_raw_zero)) / 255
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite mapped target")
        return result
