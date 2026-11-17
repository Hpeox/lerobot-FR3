# DiT architecture attribution: Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0
# See ../LICENSE-DiT.txt and ../NOTICE.md. Adaptation details below.
"""Adapted architecture; not compatible with OpenPI or official DiT checkpoints.

DiT adaLN-Zero: facebookresearch/DiT models.py (see REFERENCES.md).
Action probability path: Physical-Intelligence/openpi pi0.py, t=1 noise -> t=0 data.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from .config import ModelConfig
from .siglip import SiglipVisionEncoder


def time_embedding(t, dim):
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    z = t.float()[:, None] * freq[None]
    z = torch.cat([z.cos(), z.sin()], dim=-1)
    return F.pad(z, (0, dim - z.shape[-1]))


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class AdaLNBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(approximate="tanh"), nn.Linear(4 * width, width)
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, c):
        s1, a1, g1, s2, a2, g2 = self.modulation(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), s1, a1)
        x = x + g1[:, None] * self.attn(h, h, h, need_weights=False)[0]
        return x + g2[:, None] * self.mlp(modulate(self.norm2(x), s2, a2))


class FlowBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.self_block = AdaLNBlock(width, heads)
        self.norm = nn.LayerNorm(width)
        self.cross = nn.MultiheadAttention(width, heads, batch_first=True)

    def forward(self, x, cond, visual):
        x = self.self_block(x, cond)
        return x + self.cross(self.norm(x), visual, visual, need_weights=False)[0]


class ActionFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        w = cfg.action_width
        self.input = nn.Linear(cfg.action_dim, w)
        self.position = nn.Parameter(torch.randn(1, cfg.horizon, w) * 0.02)
        self.visual = nn.Linear(cfg.vision_width, w)
        self.state = nn.Linear(cfg.state_dim, w)
        self.task = nn.Embedding(cfg.num_tasks, w)
        self.time = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w))
        self.blocks = nn.ModuleList([FlowBlock(w, cfg.action_heads) for _ in range(cfg.action_depth)])
        self.output = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, cfg.action_dim))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def project_visual(self, visual):
        return self.visual(visual)

    def velocity(self, x, t, state, visual, tasks):
        cond = (
            self.state(state) + self.task(tasks) + self.time(time_embedding(t * 1000, self.cfg.action_width))
        )
        x = self.input(x) + self.position
        for block in self.blocks:
            x = block(x, cond, visual)
        return self.output(x)

    def loss(self, actions, valid, state, visual, tasks):
        # OpenPI direction: x_t = (1-t)*action + t*noise, target = noise-action.
        t = torch.distributions.Beta(1.5, 1.0).sample((actions.shape[0],)).to(actions.device) * 0.999 + 0.001
        noise = torch.randn_like(actions)
        x = (1 - t[:, None, None]) * actions + t[:, None, None] * noise
        pred = self.velocity(x, t, state, self.project_visual(visual), tasks)
        err = (pred - (noise - actions)).square() * valid[..., None]
        return err.sum() / (valid.sum().clamp_min(1) * self.cfg.action_dim)

    @torch.no_grad()
    def sample(self, state, visual, tasks, noise=None):
        x = (
            torch.randn(state.shape[0], self.cfg.horizon, self.cfg.action_dim, device=state.device)
            if noise is None
            else noise.clone()
        )
        visual = self.project_visual(visual)  # Once per chunk, outside Euler loop.
        dt = -1.0 / self.cfg.flow_steps
        for i in range(self.cfg.flow_steps):
            t = torch.full((state.shape[0],), 1 + i * dt, device=state.device)
            x = x + dt * self.velocity(x, t, state, visual, tasks)
        return x


class SplitPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig, initialize_pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.vision = SiglipVisionEncoder(cfg, initialize_pretrained=initialize_pretrained)
        self.action = ActionFlow(cfg)

    def forward(self, batch):
        features = self.vision.encode(batch["images"], batch["task_id"])
        flow = self.action.loss(batch["action"], batch["valid"], batch["state"], features, batch["task_id"])
        return flow, {"flow_loss": flow.detach()}

    @torch.no_grad()
    def predict_action_chunk(self, images, state, tasks):
        return self.action.sample(state, self.vision.encode(images, tasks), tasks)
