import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    image_size: int = 224
    patch_size: int = 16
    vision_backbone: str = "siglip"
    vision_model_path: str = "weights/siglip-base-patch16-224"
    vision_pretrained: bool = True
    vision_trainable_layers: int = -1
    vision_revision: str = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
    cameras: tuple[str, ...] = ("side", "top", "wrist1", "wrist2")
    vision_width: int = 768
    vision_depth: int = 12
    vision_heads: int = 12
    action_width: int = 256
    action_depth: int = 6
    action_heads: int = 8
    state_dim: int = 8
    action_dim: int = 8
    horizon: int = 16
    n_action_steps: int = 4
    num_tasks: int = 1
    flow_steps: int = 10

    def __post_init__(self):
        self.cameras = tuple(self.cameras)
        if self.vision_backbone != "siglip":
            raise ValueError("Only SigLIP is supported; old DiT checkpoints require archived code")
        if not -1 <= self.vision_trainable_layers <= self.vision_depth:
            raise ValueError("vision_trainable_layers must be -1 (all), 0 (frozen), or <= depth")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.vision_width % self.vision_heads or self.action_width % self.action_heads:
            raise ValueError("width must be divisible by heads")
        if not 0 < self.n_action_steps <= self.horizon or self.flow_steps < 1:
            raise ValueError("invalid action horizon/steps")
        if self.state_dim != 8 or self.action_dim != 8:
            raise ValueError("FR3 schema requires q[7] + gripper[1]")

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
