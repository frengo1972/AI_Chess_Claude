"""Batched, cached inference in front of :class:`~app.engine.network.ChessNet`.

MCTS spends nearly all of its time here, so this layer does three things:

* evaluates whole batches of leaves in one forward pass,
* caches results keyed by the exact encoded planes (so the cache can never
  return a value computed for a different history),
* keeps everything in ``inference_mode`` with gradients off.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from app.engine.network import ChessNet


class Evaluator:
    """Thread-unsafe by design: give each self-play worker its own instance."""

    def __init__(
        self,
        model: ChessNet,
        device: str = "cpu",
        *,
        cache_size: int = 150_000,
        use_amp: bool = False,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.cache_size = cache_size
        self.use_amp = use_amp and self.device.type == "cuda"
        self._cache: Dict[bytes, Tuple[np.ndarray, float]] = {}
        self.hits = 0
        self.misses = 0

    # -- statistics -------------------------------------------------------- #

    @property
    def cache_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0

    def clear_cache(self) -> None:
        self._cache.clear()

    # -- inference --------------------------------------------------------- #

    def evaluate(self, planes: np.ndarray) -> Tuple[np.ndarray, float]:
        return self.evaluate_many([planes])[0]

    def evaluate_many(
        self, planes_list: Sequence[np.ndarray]
    ) -> List[Tuple[np.ndarray, float]]:
        """Evaluate a batch of encoded positions.

        Returns one ``(policy_logits, value)`` pair per input, in order.
        """
        results: List[Optional[Tuple[np.ndarray, float]]] = [None] * len(planes_list)
        pending_keys: List[bytes] = []
        pending_indices: List[int] = []
        pending_planes: List[np.ndarray] = []

        for i, planes in enumerate(planes_list):
            key = planes.tobytes()
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
                self.hits += 1
            else:
                self.misses += 1
                pending_keys.append(key)
                pending_indices.append(i)
                pending_planes.append(planes)

        if pending_planes:
            batch = torch.from_numpy(np.stack(pending_planes)).to(
                self.device, non_blocking=True
            )
            with torch.inference_mode():
                if self.use_amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits, values = self.model(batch)
                    logits = logits.float()
                    values = values.float()
                else:
                    logits, values = self.model(batch)
            logits_np = logits.cpu().numpy()
            values_np = values.cpu().numpy()

            if len(self._cache) > self.cache_size:
                self._cache.clear()

            for slot, key, index in zip(
                range(len(pending_indices)), pending_keys, pending_indices
            ):
                entry = (logits_np[slot], float(values_np[slot]))
                self._cache[key] = entry
                results[index] = entry

        return results  # type: ignore[return-value]


def masked_softmax(logits: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Softmax restricted to ``indices`` (the legal moves).

    This is where the rules of chess meet the network: the raw 4672-way policy
    is *never* used directly, only its restriction to the legal move set
    produced by ``python-chess``.
    """
    selected = logits[list(indices)]
    selected = selected - selected.max()
    exp = np.exp(selected)
    total = exp.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(indices), 1.0 / max(len(indices), 1), dtype=np.float32)
    return (exp / total).astype(np.float32)
