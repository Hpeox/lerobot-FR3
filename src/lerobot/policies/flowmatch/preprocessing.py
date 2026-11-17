"""Training-compatible SigLIP preprocessing and native-unit normalization.

No HDF5, dataset conversion, camera access, or hardware imports.
"""

from functools import lru_cache

import torch


@lru_cache(maxsize=8)
def image_processor(size: int):
    # Transformers5 defaults to TorchvisionBackend, while training4.57.1 used
    # PIL. Use the official PIL class explicitly to preserve camera pixel values.
    from transformers.models.siglip.image_processing_pil_siglip import SiglipImageProcessorPil

    return SiglipImageProcessorPil(size={"height": size, "width": size})


def preprocess_rgb(value: torch.Tensor, size: int) -> torch.Tensor:
    """BCHW RGB uint8 / float [0,1] -> official SigLIP pixels, once only."""
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError("RGB must be BCHW")
    device = value.device
    if value.dtype != torch.uint8:
        if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError("RGB must be uint8 or float [0,1], before SigLIP normalization")
        value = (value * 255).round().to(torch.uint8)
    pixels = image_processor(size)(images=list(value.cpu()), return_tensors="pt")["pixel_values"]
    return pixels.to(device)


class Normalizer:
    def __init__(self, stats: dict):
        self.stats = stats
        for kind in ("state", "action"):
            mean, std = (torch.as_tensor(stats[kind][name], dtype=torch.float32) for name in ("mean", "std"))
            if mean.shape != (8,) or std.shape != (8,) or not torch.isfinite(mean).all():
                raise ValueError("Expected eight finite state/action normalization values")
            if not torch.isfinite(std).all() or (std <= 0).any():
                raise ValueError("Normalization standard deviations must be finite and positive")

    def normalize(self, value, kind):
        mean = torch.as_tensor(self.stats[kind]["mean"], device=value.device, dtype=value.dtype)
        std = torch.as_tensor(self.stats[kind]["std"], device=value.device, dtype=value.dtype)
        return (value - mean) / std

    def denormalize(self, value, kind):
        mean = torch.as_tensor(self.stats[kind]["mean"], device=value.device, dtype=value.dtype)
        std = torch.as_tensor(self.stats[kind]["std"], device=value.device, dtype=value.dtype)
        return value * std + mean
