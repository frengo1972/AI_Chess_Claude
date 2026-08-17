"""The policy/value residual network.

A single tower of residual blocks feeds two heads:

* **policy** -- 4672 logits, one per (from-square, move-plane) pair. Illegal
  moves are masked *outside* the network, by :mod:`app.engine.encoding`.
* **value**  -- one scalar in ``[-1, 1]``: the expected result of the game from
  the point of view of the side to move.

This is the AlphaZero architecture, shrunk. The defaults (6 blocks x 96
filters) train meaningfully within hours on a laptop GPU; ``config.PRESETS``
scales it up without touching this file.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import NetworkConfig
from app.engine.encoding import NUM_MOVE_PLANES, POLICY_SIZE, num_input_planes


class SqueezeExcitation(nn.Module):
    """Channel-wise gating, as used by Leela Chess Zero's residual blocks."""

    def __init__(self, channels: int, ratio: int) -> None:
        super().__init__()
        hidden = max(channels // ratio, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3))
        hidden = F.relu(self.fc1(pooled))
        scale_shift = self.fc2(hidden)
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = torch.sigmoid(scale).unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return x * scale + shift


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, se_ratio: int = 0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se: Optional[SqueezeExcitation] = (
            SqueezeExcitation(channels, se_ratio) if se_ratio > 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        return F.relu(out + residual)


class ChessNet(nn.Module):
    """Policy/value network over the AlphaZero chess encoding."""

    def __init__(self, config: Optional[NetworkConfig] = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        in_planes = num_input_planes(self.config.history_length)
        filters = self.config.filters

        self.input_conv = nn.Conv2d(in_planes, filters, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(filters)
        self.tower = nn.Sequential(
            *[
                ResidualBlock(filters, self.config.se_ratio)
                for _ in range(self.config.residual_blocks)
            ]
        )

        policy_filters = self.config.policy_head_filters
        self.policy_conv = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(filters)
        self.policy_out = nn.Conv2d(filters, policy_filters, 1)
        self.policy_projection: Optional[nn.Linear] = (
            None
            if policy_filters == NUM_MOVE_PLANES
            else nn.Linear(policy_filters * 64, POLICY_SIZE)
        )

        self.value_conv = nn.Conv2d(filters, 32, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 64, self.config.value_hidden)
        self.value_fc2 = nn.Linear(self.config.value_hidden, 1)

        self._initialise()

    def _initialise(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(policy_logits[B, 4672], value[B])``."""
        out = F.relu(self.input_bn(self.input_conv(x)))
        out = self.tower(out)

        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = self.policy_out(policy)
        policy = policy.flatten(1)  # (B, planes*64) == plane*64 + square
        if self.policy_projection is not None:
            policy = self.policy_projection(policy)

        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.flatten(1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)

        return policy, value

    # -- introspection ----------------------------------------------------- #

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def size_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters()) + sum(
            b.numel() * b.element_size() for b in self.buffers()
        )

    def describe(self) -> dict:
        return {
            "history_length": self.config.history_length,
            "input_planes": num_input_planes(self.config.history_length),
            "residual_blocks": self.config.residual_blocks,
            "filters": self.config.filters,
            "se_ratio": self.config.se_ratio,
            "parameters": self.parameter_count(),
            "size_mb": round(self.size_bytes() / (1024 * 1024), 3),
            "policy_size": POLICY_SIZE,
        }


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #


def save_checkpoint(
    path,
    model: ChessNet,
    *,
    iteration: int = 0,
    metadata: Optional[dict] = None,
    optimizer_state: Optional[dict] = None,
) -> None:
    from pathlib import Path

    payload = {
        "format": 1,
        "network_config": vars(model.config),
        "state_dict": model.state_dict(),
        "iteration": iteration,
        "metadata": metadata or {},
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)


def load_checkpoint(path, device: str = "cpu") -> Tuple[ChessNet, dict]:
    """Rebuild a :class:`ChessNet` from disk together with its metadata."""
    payload = torch.load(path, map_location=device, weights_only=False)
    config = NetworkConfig(**payload["network_config"])
    model = ChessNet(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    info = {
        "iteration": payload.get("iteration", 0),
        "metadata": payload.get("metadata", {}),
        "network_config": payload["network_config"],
    }
    return model, info


def resolve_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    return "cuda" if torch.cuda.is_available() else "cpu"
