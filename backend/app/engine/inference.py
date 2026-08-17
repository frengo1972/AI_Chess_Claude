"""One GPU serving every self-play worker, instead of a small model per CPU core.

The motivation is measured: a single CPU thread evaluates ~150 positions/s and
**gains nothing from batching** (73/s at batch 1, 170/s at batch 512 -- one
thread is compute-bound), while the same network on the GPU does ~1.000/s at
batch 16 and ~13.000/s at batch 256. Self-play spends 87% of its time inside the
network. So the workers stop owning a model and become *clients*: they send
their leaves here, and this server merges every worker's request into one
forward pass, the way Lc0 and KataGo do.

**What it is actually worth, measured on this project** (RTX 2000 Ada laptop,
22 logical cores, 12 self-play games):

===========================================  =========  =============
configuration                                 speed-up   mean batch
===========================================  =========  =============
``tiny`` 3x64, 12 workers, 16 leaves            0.84x        61
``small`` 6x96, 8 workers, 16 leaves            0.88x        27
``small`` 6x96, 16 workers, 48 leaves           1.55x       113
``small`` 6x96, 22 workers, 48 leaves           1.40x       120
===========================================  =========  =============

Far below what the raw throughput numbers suggest, and worth understanding
before turning it on:

* **A worker has one request in flight at a time.** It cannot produce the next
  batch of leaves until the current one comes back, so the server sees a trickle
  rather than a flood and the batch stays small. Widening the collection window
  does not help: it just adds that latency to every worker. What helps is more
  leaves per request (``search.max_batch_size``), which is why the third row is
  the only clear win -- and that trades search quality, since 48 leaves in
  flight out of 96 simulations means heavy virtual loss.
* **The policy is wide.** Each answer carries 4672 float32 logits (18.7 KB per
  position); a 12-game batch moves ~2.4 GB through the queues. On a small
  network that costs more than the forward pass it saves.

The way past both is what the strong engines do: many *concurrent games per
worker*, so leaves from independent trees fill one request and latency is
hidden, plus answers restricted to the legal moves. Until that exists this stays
off by default -- ``inference.enabled`` in the presets.

Design notes:

* **The server is a thread, not a process.** It runs inside whoever calls
  ``generate_selfplay`` -- the trainer -- which is idle waiting on futures
  anyway. PyTorch releases the GIL during the forward pass and ``Queue.get``
  releases it while waiting, so a thread is enough, and it saves a process plus
  a checkpoint reload.
* **Queues are handed to workers through the pool initialiser.** A
  ``multiprocessing.Queue`` cannot be pickled into a ``submit()`` argument; it
  can only cross a process boundary while that process is being created. Each
  worker claims one response queue via a shared counter, so no two workers ever
  read each other's answers.
* **Workers no longer load the checkpoint at all**, which also removes one model
  deserialisation per worker per iteration.
* **A failure degrades, never hangs.** Clients wait with a timeout and raise;
  self-play falls back to local CPU evaluators if the server cannot start.

Like everything under ``app/engine``, this module never reaches the classical
engine: it moves the network's own output between processes and nothing else.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from app.engine.evaluator import BaseEvaluator
from app.engine.network import load_checkpoint

REQUEST_TIMEOUT_SECONDS = 120.0
"""How long a worker waits for its answer before deciding the server is gone.
Generous: the server may be working through a large queued batch."""

_SHUTDOWN = None
"""Sentinel pushed on the request queue to wake the server up for shutdown."""


# --------------------------------------------------------------------------- #
# Worker side
# --------------------------------------------------------------------------- #

_CLIENT: Optional["RemoteEvaluator"] = None


class RemoteEvaluator(BaseEvaluator):
    """A worker's view of the shared GPU: same interface, no model of its own."""

    def __init__(
        self,
        client_id: int,
        request_queue: Any,
        response_queue: Any,
        *,
        cache_size: int = 150_000,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(cache_size=cache_size)
        self.client_id = client_id
        self.requests = request_queue
        self.responses = response_queue
        self.timeout = timeout
        self._next_id = 0
        self.waited_seconds = 0.0
        """Time spent blocked on the server; the honest cost of going remote."""

    def forward(self, planes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._next_id += 1
        request_id = self._next_id
        started = time.perf_counter()
        self.requests.put((self.client_id, request_id, planes))

        while True:
            try:
                answer_id, logits, values = self.responses.get(timeout=self.timeout)
            except queue.Empty as error:
                raise RuntimeError(
                    "the inference server stopped answering "
                    f"(client {self.client_id}, waited {self.timeout:.0f}s)"
                ) from error
            if answer_id == request_id:
                self.waited_seconds += time.perf_counter() - started
                return logits, values
            # A stale answer from an abandoned request: drop it and keep waiting.


def worker_initializer(
    request_queue: Any, response_queues: Sequence[Any], counter: Any
) -> None:
    """Claim one response queue for this process, once, at pool start-up."""
    global _CLIENT
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    _CLIENT = RemoteEvaluator(index, request_queue, response_queues[index])


def worker_evaluator() -> Optional[RemoteEvaluator]:
    """The evaluator claimed by :func:`worker_initializer`, if there is one."""
    return _CLIENT


# --------------------------------------------------------------------------- #
# Server side
# --------------------------------------------------------------------------- #


@dataclass
class ServerStats:
    batches: int = 0
    positions: int = 0
    forward_seconds: float = 0.0
    idle_seconds: float = 0.0

    @property
    def mean_batch(self) -> float:
        return self.positions / self.batches if self.batches else 0.0

    @property
    def positions_per_second(self) -> float:
        return self.positions / self.forward_seconds if self.forward_seconds else 0.0

    def to_dict(self) -> dict:
        return {
            "batches": self.batches,
            "positions": self.positions,
            "mean_batch": round(self.mean_batch, 1),
            "positions_per_second": round(self.positions_per_second, 1),
            "forward_seconds": round(self.forward_seconds, 2),
            "idle_seconds": round(self.idle_seconds, 2),
        }


class InferenceServer:
    """Merges every worker's leaves into one forward pass.

    The collection window is the only tuning knob that matters: waiting a
    millisecond or two lets the other workers' requests arrive, which is exactly
    what turns eight small batches into one large one. Waiting longer than a
    worker takes to produce its next batch just adds latency.
    """

    def __init__(
        self,
        checkpoint,
        *,
        device: str = "cuda",
        max_batch: int = 256,
        collect_timeout_ms: float = 2.0,
        use_amp: bool = False,
        request_queue: Any = None,
        response_queues: Sequence[Any] = (),
        counter: Any = None,
    ) -> None:
        import torch

        self.max_batch = max(1, int(max_batch))
        self.collect_timeout = max(0.0, collect_timeout_ms / 1000.0)
        self.requests = request_queue
        self.responses = list(response_queues)
        self.counter = counter
        """Shared claim counter: each worker takes one response queue with it."""
        self.stats = ServerStats()

        self.model, _ = load_checkpoint(checkpoint, device=device)
        self.model.eval()
        self.device = torch.device(device)
        self.use_amp = use_amp and self.device.type == "cuda"

        self._torch = torch
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> "InferenceServer":
        self._thread = threading.Thread(
            target=self.serve, name="inference-server", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, timeout: float = 10.0) -> ServerStats:
        self._stop.set()
        if self.requests is not None:
            try:
                self.requests.put(_SHUTDOWN)
            except Exception:  # noqa: BLE001 - the queue may already be closed
                pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        return self.stats

    # -- loop --------------------------------------------------------------- #

    def serve(self) -> None:
        while not self._stop.is_set():
            batch = self._collect()
            if batch:
                self._answer(batch)

    def _collect(self) -> List[Tuple[int, int, np.ndarray]]:
        """Take one request, then keep taking until the batch or the window fills."""
        idle_started = time.perf_counter()
        try:
            first = self.requests.get(timeout=0.2)
        except queue.Empty:
            self.stats.idle_seconds += time.perf_counter() - idle_started
            return []
        self.stats.idle_seconds += time.perf_counter() - idle_started
        if first is _SHUTDOWN:
            self._stop.set()
            return []

        batch = [first]
        total = len(first[2])
        deadline = time.perf_counter() + self.collect_timeout
        while total < self.max_batch:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = self.requests.get(timeout=remaining)
            except queue.Empty:
                break
            if item is _SHUTDOWN:
                self._stop.set()
                break
            batch.append(item)
            total += len(item[2])
        return batch

    def _answer(self, batch: List[Tuple[int, int, np.ndarray]]) -> None:
        torch = self._torch
        planes = np.concatenate([item[2] for item in batch])
        started = time.perf_counter()
        tensor = torch.from_numpy(planes).to(self.device, non_blocking=True)
        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits, values = self.model(tensor)
                logits = logits.float()
                values = values.float()
            else:
                logits, values = self.model(tensor)
        logits_np = logits.cpu().numpy()
        values_np = values.cpu().numpy()
        self.stats.forward_seconds += time.perf_counter() - started
        self.stats.batches += 1
        self.stats.positions += len(planes)

        offset = 0
        for client_id, request_id, request_planes in batch:
            count = len(request_planes)
            self.responses[client_id].put(
                (
                    request_id,
                    logits_np[offset : offset + count],
                    values_np[offset : offset + count],
                )
            )
            offset += count
