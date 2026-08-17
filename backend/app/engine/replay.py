"""Replay buffer for self-play training data.

Storing raw ``(14*T + 7, 8, 8)`` float32 tensors would cost ~16 KB per
position, so a 200k-position buffer would need 3 GB. Two properties of the
encoding make a ~30x compression trivial and lossless:

* the ``14*T`` history planes are strictly binary -> bit-pack them;
* the 7 trailing planes are constant across the board -> store 7 scalars.

A sample therefore costs a few hundred bytes, and unpacking is a couple of
numpy calls per batch.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from app.engine.encoding import (
    CONSTANT_PLANES,
    PLANES_PER_STEP,
    POLICY_SIZE,
    num_input_planes,
)


@dataclass(slots=True)
class Sample:
    """One training example: a position, its search policy, and its outcome."""

    bits: np.ndarray  # uint8, packed binary history planes
    scalars: np.ndarray  # float32[CONSTANT_PLANES] -- exact, only 28 bytes
    policy_indices: np.ndarray  # int32
    policy_probs: np.ndarray  # float16
    value: float  # game outcome from the side-to-move's perspective


def pack_planes(planes: np.ndarray, history_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split an encoded position into (packed history bits, constant scalars)."""
    split = PLANES_PER_STEP * history_length
    history = planes[:split]
    constants = planes[split:]
    bits = np.packbits(history.astype(np.uint8).reshape(-1))
    # The trailing planes are board-constant but not binary (clocks are
    # normalised counters), so they stay float32: 28 bytes, exactly round-trips.
    scalars = np.ascontiguousarray(constants[:, 0, 0], dtype=np.float32)
    return bits, scalars


def unpack_planes(
    bits: np.ndarray, scalars: np.ndarray, history_length: int
) -> np.ndarray:
    """Inverse of :func:`pack_planes`."""
    split = PLANES_PER_STEP * history_length
    total = num_input_planes(history_length)
    out = np.empty((total, 8, 8), dtype=np.float32)
    flat = np.unpackbits(bits, count=split * 64).astype(np.float32)
    out[:split] = flat.reshape(split, 8, 8)
    out[split:] = scalars.astype(np.float32).reshape(CONSTANT_PLANES, 1, 1)
    return out


def make_sample(
    planes: np.ndarray,
    policy: dict,
    value: float,
    history_length: int,
) -> Sample:
    bits, scalars = pack_planes(planes, history_length)
    indices = np.fromiter(policy.keys(), dtype=np.int32, count=len(policy))
    probs = np.fromiter(policy.values(), dtype=np.float32, count=len(policy))
    total = probs.sum()
    if total > 0:
        probs = probs / total
    return Sample(bits, scalars, indices, probs.astype(np.float16), float(value))


class ReplayBuffer:
    """Fixed-size FIFO of :class:`Sample`, safe for one writer + one reader."""

    def __init__(self, capacity: int, history_length: int) -> None:
        self.capacity = capacity
        self.history_length = history_length
        self._items: Deque[Sample] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.total_added = 0

    def __len__(self) -> int:
        return len(self._items)

    def extend(self, samples: Iterable[Sample]) -> None:
        with self._lock:
            for sample in samples:
                self._items.append(sample)
                self.total_added += 1

    def sample_batch(
        self,
        batch_size: int,
        rng: np.random.Generator,
        recent_fraction: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw a batch, returning ``(planes, dense_policy, values)``."""
        with self._lock:
            size = len(self._items)
            if size == 0:
                raise ValueError("replay buffer is empty")
            window = size if recent_fraction >= 1.0 else max(
                batch_size, int(size * recent_fraction)
            )
            window = min(window, size)
            offset = size - window
            picks = rng.integers(offset, size, size=batch_size)
            chosen = [self._items[int(i)] for i in picks]

        planes = np.stack(
            [unpack_planes(s.bits, s.scalars, self.history_length) for s in chosen]
        )
        policies = np.zeros((batch_size, POLICY_SIZE), dtype=np.float32)
        for row, sample in enumerate(chosen):
            policies[row, sample.policy_indices] = sample.policy_probs.astype(np.float32)
        values = np.asarray([s.value for s in chosen], dtype=np.float32)
        return planes, policies, values

    # -- persistence ------------------------------------------------------- #

    def save(self, path: Path) -> None:
        with self._lock:
            items = list(self._items)
        if not items:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        lengths = np.asarray([len(s.policy_indices) for s in items], dtype=np.int32)
        np.savez_compressed(
            path,
            history_length=np.int32(self.history_length),
            bits=np.stack([s.bits for s in items]),
            scalars=np.stack([s.scalars for s in items]),
            lengths=lengths,
            policy_indices=np.concatenate([s.policy_indices for s in items]),
            policy_probs=np.concatenate([s.policy_probs for s in items]),
            values=np.asarray([s.value for s in items], dtype=np.float16),
        )

    def load(self, path: Path) -> int:
        data = np.load(path)
        lengths = data["lengths"]
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        all_indices = data["policy_indices"]
        all_probs = data["policy_probs"]
        bits = data["bits"]
        scalars = data["scalars"]
        values = data["values"]
        samples = [
            Sample(
                bits[i],
                scalars[i],
                all_indices[offsets[i] : offsets[i + 1]],
                all_probs[offsets[i] : offsets[i + 1]],
                float(values[i]),
            )
            for i in range(len(lengths))
        ]
        self.extend(samples)
        return len(samples)

    def load_directory(self, directory: Path, limit_files: Optional[int] = None) -> int:
        files = sorted(Path(directory).glob("*.npz"))
        if limit_files:
            files = files[-limit_files:]
        loaded = 0
        for file in files:
            try:
                loaded += self.load(file)
            except Exception:  # a truncated shard must not kill a resumed run
                continue
        return loaded


def samples_from_game(
    encoded_positions: Sequence[np.ndarray],
    policies: Sequence[dict],
    values: Sequence[float],
    history_length: int,
) -> List[Sample]:
    return [
        make_sample(planes, policy, value, history_length)
        for planes, policy, value in zip(encoded_positions, policies, values)
    ]
