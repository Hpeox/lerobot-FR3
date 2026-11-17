"""Official Transformers SigLIP vision backbone; no custom visual transformer.

Official weights: https://huggingface.co/google/siglip-base-patch16-224
Inputs use the official processor before reaching this module.
"""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
from transformers import SiglipVisionConfig, SiglipVisionModel


class SiglipVisionEncoder(nn.Module):
    def __init__(self, cfg, initialize_pretrained=True):
        super().__init__()
        self.cfg = cfg
        if cfg.vision_pretrained and initialize_pretrained:
            source = Path(cfg.vision_model_path)
            if not (source / "config.json").exists():
                raise FileNotFoundError(
                    "Official SigLIP snapshot missing. Run scripts/download_siglip.py first; no random fallback."
                )
            provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
            if (
                provenance["repo_id"] != "google/siglip-base-patch16-224"
                or provenance["revision"] != cfg.vision_revision
            ):
                raise ValueError("Pretrained SigLIP provenance/revision mismatch")
            self.backbone = SiglipVisionModel.from_pretrained(
                source, local_files_only=True, attn_implementation="sdpa"
            )
        else:
            # Offline construction for checkpoint restoration or explicit tiny unit tests.
            config = SiglipVisionConfig(
                hidden_size=cfg.vision_width,
                intermediate_size=4 * cfg.vision_width,
                num_hidden_layers=cfg.vision_depth,
                num_attention_heads=cfg.vision_heads,
                image_size=cfg.image_size,
                patch_size=cfg.patch_size,
            )
            self.backbone = SiglipVisionModel(config)
        actual = self.backbone.config
        if (
            actual.hidden_size,
            actual.num_hidden_layers,
            actual.num_attention_heads,
            actual.image_size,
            actual.patch_size,
        ) != (cfg.vision_width, cfg.vision_depth, cfg.vision_heads, cfg.image_size, cfg.patch_size):
            raise ValueError("SigLIP snapshot/config dimensions mismatch")
        self.camera = nn.Embedding(len(cfg.cameras), cfg.vision_width)
        nn.init.zeros_(self.camera.weight)
        self.backbone.vision_model.use_head = False  # Spatial tokens only; skip unused pooler compute.
        self.set_finetuning(cfg.vision_trainable_layers)

    def set_finetuning(self, layers):
        self.backbone.requires_grad_(layers == -1)
        if layers > 0:
            for layer in self.backbone.vision_model.encoder.layers[-layers:]:
                layer.requires_grad_(True)
            self.backbone.vision_model.post_layernorm.requires_grad_(True)
        # Attention pooler is not used: consume the official spatial patch states.
        if hasattr(self.backbone.vision_model, "head"):
            self.backbone.vision_model.head.requires_grad_(False)
        self.camera.requires_grad_(layers != 0)

    def encode(self, images, tasks=None):
        b, v, c, h, w = images.shape
        if (v, c, h, w) != (len(self.cfg.cameras), 3, self.cfg.image_size, self.cfg.image_size):
            raise ValueError("Expected B,V,3,H,W in configured camera order")
        # SigLIP already applies its trained post-layer norm to last_hidden_state.
        x = self.backbone(pixel_values=images.reshape(b * v, c, h, w)).last_hidden_state
        grid = self.cfg.image_size // self.cfg.patch_size
        x = x.transpose(1, 2).reshape(b * v, self.cfg.vision_width, grid, grid)
        x = F.adaptive_avg_pool2d(x, (2, 2)).flatten(2).transpose(1, 2)
        camera = self.camera(torch.arange(v, device=images.device)).repeat(b, 1)
        x = x + camera[:, None, :]
        return x.reshape(b, v * 4, self.cfg.vision_width)
