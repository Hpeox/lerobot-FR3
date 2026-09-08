"""Small, deployment-safe adapter for timm's DINOv2-S/14 model.

The adapter deliberately imports timm lazily.  Training environments can use
the optional ``timm-dep`` extra, while unit tests and checkpoint inspection do
not need to import or download a vision model.  A loaded policy checkpoint
contains the complete DINO state dict, so deployment constructs the same
architecture with ``pretrained=False`` and never reaches the network.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


class DINOv2SpatialBackbone(nn.Module):
    """Return final normalized DINOv2 patch tokens without the CLS token."""

    def __init__(
        self,
        model_name: str,
        *,
        image_size: tuple[int, int] = (336, 448),
        patch_size: int = 14,
        token_dim: int = 384,
        pretrained: bool = True,
        checkpoint: str | None = None,
        require_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "ACMT-ACTv2 requires timm for DINOv2-S/14. Install the optional "
                "'timm-dep' extra before constructing the policy."
            ) from exc

        self.model_name = str(model_name)
        self.image_size = tuple(int(value) for value in image_size)
        self.patch_size = int(patch_size)
        self.token_dim = int(token_dim)
        if any(value <= 0 for value in self.image_size) or any(value % self.patch_size for value in self.image_size):
            raise ValueError(f"DINOv2 image size must be divisible by patch size: {self.image_size}")

        # ``dynamic_img_size`` keeps timm from silently resizing/cropping a
        # camera frame.  The policy processor already made the exact 336x448
        # input and the runtime checks the shape below.
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=bool(pretrained and checkpoint is None),
            num_classes=0,
            img_size=self.image_size,
            dynamic_img_size=True,
        )
        if checkpoint is not None:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_file():
                if require_checkpoint:
                    raise FileNotFoundError(f"DINOv2 checkpoint not found: {checkpoint_path}")
            else:
                self._load_checkpoint(checkpoint_path)
        elif require_checkpoint and not pretrained:
            raise FileNotFoundError("DINOv2 requires a local checkpoint when pretrained=False")

        actual_dim = int(getattr(self.backbone, "num_features", self.token_dim))
        if actual_dim != self.token_dim:
            raise ValueError(f"DINOv2 model reports {actual_dim} features, expected {self.token_dim}")
        self._freeze()

    def _load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, Mapping):
            for key in ("state_dict", "model", "model_state_dict"):
                candidate = payload.get(key)
                if isinstance(candidate, Mapping):
                    payload = candidate
                    break
        if not isinstance(payload, Mapping):
            raise ValueError(f"DINOv2 checkpoint must contain a state-dict mapping: {path}")
        state: dict[str, Tensor] = {}
        for key, value in payload.items():
            if not isinstance(value, Tensor):
                continue
            clean = str(key)
            for prefix in ("module.", "backbone.", "model."):
                if clean.startswith(prefix):
                    clean = clean[len(prefix) :]
            state[clean] = value
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        # Classification heads and training-only projection keys are harmless,
        # but silently missing backbone blocks would make a deployment model
        # unusable.  Check the common backbone prefix rather than requiring
        # optional timm classifier keys to exist.
        missing_backbone = [key for key in missing if not key.startswith(("head.", "fc.", "classifier."))]
        if missing_backbone:
            raise RuntimeError(
                f"DINOv2 checkpoint {path} is missing {len(missing_backbone)} backbone keys; "
                f"first={missing_backbone[:3]}"
            )
        if unexpected:
            # Keep this informational: timm checkpoints commonly contain a
            # pretraining head that is intentionally not instantiated here.
            print(
                f"[acmt_actv2] ignored {len(unexpected)} DINOv2 checkpoint keys from {path.name}",
                flush=True,
            )

    def _freeze(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        # The policy's outer train() must not put DINO's dropout/stochastic
        # depth or normalization layers into training mode.
        super().train(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, *self.image_size):
            raise ValueError(
                "DINOv2 ACMT-ACT expects normalized RGB [B,3,336,448], "
                f"got {tuple(image.shape)}"
            )
        features: Any = self.backbone.forward_features(image)
        if isinstance(features, Mapping):
            tokens = features.get("x_norm_patchtokens")
            if tokens is None:
                tokens = features.get("x_prenorm")
                if isinstance(tokens, Tensor) and hasattr(self.backbone, "norm"):
                    # Some timm versions expose only the pre-normalization
                    # token sequence.  Match DINOv2's final normalized patch
                    # tokens before removing the prefix token.
                    tokens = self.backbone.norm(tokens)
        else:
            tokens = features
        if not isinstance(tokens, Tensor) or tokens.ndim != 3:
            raise RuntimeError("timm DINOv2 forward_features did not return a [B,N,C] token tensor")
        if tokens.shape[-1] != self.token_dim:
            raise RuntimeError(
                f"DINOv2 token width is {tokens.shape[-1]}, expected {self.token_dim}"
            )
        expected = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)
        if tokens.shape[1] != expected:
            prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))
            if tokens.shape[1] == expected + prefix:
                tokens = tokens[:, prefix:]
            else:
                raise RuntimeError(
                    f"DINOv2 returned {tokens.shape[1]} tokens, expected {expected} patch tokens"
                )
        return tokens


__all__ = ["DINOv2SpatialBackbone"]
