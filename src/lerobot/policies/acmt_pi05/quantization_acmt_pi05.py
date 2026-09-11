#!/usr/bin/env python

"""Selective bitsandbytes INT8 support for ACMT-PI05 deployment.

The training checkpoint is deliberately left untouched.  Quantization is
performed after a strict CPU load and before the model is moved to CUDA.
Only large transformer Linear layers are eligible; input/output adapters,
normalization, embeddings, and gripper-sensitive heads remain floating point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
from torch import nn


DEFAULT_MIN_NUMEL = 4_000_000
DEFAULT_THRESHOLD = 6.0

_EXCLUDED_KEYWORDS = (
    "embed_tokens",
    "embedding",
    "lm_head",
    "action_in",
    "action_out",
    "action_input",
    "action_output",
    "action_head",
    "state_proj",
    "state_projection",
    "gripper",
    "norm",
)

_STAGE_MARKERS = {
    "language": ("paligemma_with_expert.paligemma.model.language_model.layers.",),
    "action": ("paligemma_with_expert.gemma_expert.model.layers.",),
    "vision": ("paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.",),
}


@dataclass(frozen=True)
class QuantizedLinearRecord:
    name: str
    shape: tuple[int, ...]
    numel: int
    stage: str


@dataclass(frozen=True)
class QuantizationReport:
    backend: str
    stages: tuple[str, ...]
    min_numel: int
    threshold: float
    records: tuple[QuantizedLinearRecord, ...]

    @property
    def layer_count(self) -> int:
        return len(self.records)

    @property
    def parameter_count(self) -> int:
        return sum(record.numel for record in self.records)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["stages"] = list(self.stages)
        result["records"] = [asdict(record) for record in self.records]
        result["layer_count"] = self.layer_count
        result["parameter_count"] = self.parameter_count
        return result


def _stage_for_name(name: str, stages: Iterable[str]) -> str | None:
    for stage in stages:
        if any(marker in name for marker in _STAGE_MARKERS[stage]):
            return stage
    return None


def should_quantize_linear(
    name: str,
    module: nn.Linear,
    *,
    stage: str,
    min_numel: int = DEFAULT_MIN_NUMEL,
) -> bool:
    """Return whether a Linear belongs to the requested conservative allowlist."""

    if stage not in _STAGE_MARKERS:
        raise ValueError(f"unknown ACMT-PI05 quantization stage: {stage!r}")
    if not isinstance(module, nn.Linear) or module.weight.numel() < min_numel:
        return False
    lowered = name.lower()
    if any(keyword in lowered for keyword in _EXCLUDED_KEYWORDS):
        return False
    return any(marker in name for marker in _STAGE_MARKERS[stage])


def _replace_child(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, attribute = name.rsplit(".", 1)
    parent = root.get_submodule(parent_name)
    setattr(parent, attribute, replacement)


def _make_int8_linear(module: nn.Linear, *, threshold: float) -> nn.Module:
    try:
        import bitsandbytes as bnb
    except ImportError as exc:  # pragma: no cover - exercised in environments without the optional extra
        raise RuntimeError(
            "Selective ACMT-PI05 INT8 requires bitsandbytes; install the deployment extra first"
        ) from exc

    if module.weight.device.type != "cpu":
        raise ValueError("ACMT-PI05 INT8 replacement must happen before moving weights to CUDA")

    replacement = bnb.nn.Linear8bitLt(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        has_fp16_weights=False,
        threshold=threshold,
    )
    # Int8Params retains the floating-point source on CPU and materializes
    # bitsandbytes CB/SCB state when the module is moved to CUDA.
    replacement.weight = bnb.nn.Int8Params(
        module.weight.detach().contiguous(),
        requires_grad=False,
        has_fp16_weights=False,
    )
    if module.bias is not None:
        replacement.bias = nn.Parameter(
            module.bias.detach().to(dtype=module.weight.dtype).contiguous(),
            requires_grad=False,
        )
    return replacement


def quantize_acmt_pi05_int8(
    root: nn.Module,
    *,
    stages: Iterable[str],
    min_numel: int = DEFAULT_MIN_NUMEL,
    threshold: float = DEFAULT_THRESHOLD,
) -> QuantizationReport:
    """Replace eligible Linear modules in ``root`` and return an audit report."""

    stages = tuple(dict.fromkeys(stages))
    for stage in stages:
        if stage not in _STAGE_MARKERS:
            raise ValueError(f"unknown ACMT-PI05 quantization stage: {stage!r}")
    if not stages:
        raise ValueError("at least one ACMT-PI05 quantization stage is required")
    if any(parameter.device.type != "cpu" for parameter in root.parameters()):
        raise ValueError("ACMT-PI05 INT8 quantization requires a CPU-resident model")

    candidates: list[tuple[str, nn.Linear, str]] = []
    for name, module in root.named_modules():
        stage = _stage_for_name(name, stages)
        if stage is not None and should_quantize_linear(
            name=name, module=module, stage=stage, min_numel=min_numel
        ):
            candidates.append((name, module, stage))

    records: list[QuantizedLinearRecord] = []
    for name, module, stage in candidates:
        replacement = _make_int8_linear(module, threshold=threshold)
        _replace_child(root, name, replacement)
        records.append(
            QuantizedLinearRecord(
                name=name,
                shape=tuple(module.weight.shape),
                numel=module.weight.numel(),
                stage=stage,
            )
        )

    return QuantizationReport(
        backend="bitsandbytes_int8",
        stages=stages,
        min_numel=min_numel,
        threshold=threshold,
        records=tuple(records),
    )


def linear_compute_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    """Return a usable activation dtype for float or bitsandbytes Linear modules."""

    weight = getattr(module, "weight", None)
    dtype = getattr(weight, "dtype", None)
    # Int8Params reports torch.int8 after CUDA materialization, but the
    # bitsandbytes kernel consumes floating-point activations.
    if dtype in (torch.int8, torch.uint8) or module.__class__.__module__.startswith("bitsandbytes"):
        return fallback
    return dtype if dtype is not None else fallback


__all__ = [
    "DEFAULT_MIN_NUMEL",
    "DEFAULT_THRESHOLD",
    "QuantizationReport",
    "QuantizedLinearRecord",
    "linear_compute_dtype",
    "quantize_acmt_pi05_int8",
    "should_quantize_linear",
]
