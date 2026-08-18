"""Self-play: the only source of training data in this project.

The network plays both sides. For every position we store

* the encoded position,
* the search policy (visit-count distribution, or the raw prior when
  ``simulations == 1``),
* the final result of the game, from that position's side-to-move perspective.

Games are generated in worker processes. Workers receive a *checkpoint path*
rather than a live model, because Windows uses ``spawn`` and CUDA tensors do
not survive process boundaries. Each worker keeps a module-level model cache so
a checkpoint is deserialised once per process, not once per game.

A worker can also publish the game it is playing for a spectator, through
:mod:`app.engine.watch`. That is a one-way tap: nothing published there comes
back into the samples, and with watching switched off it costs one small file
read every half second.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.pgn
import numpy as np

from app.config import RunConfig
from app.engine.encoding import PositionHistory
from app.engine.evaluator import BaseEvaluator, Evaluator
from app.engine.inference import InferenceServer, worker_evaluator, worker_initializer
from app.engine.mcts import MCTS
from app.engine.network import load_checkpoint, resolve_device
from app.engine.replay import Sample, make_sample
from app.engine.rules import result_string, terminal_state
from app.engine.watch import WatchPublisher


@dataclass
class GameRecord:
    samples: List[Sample] = field(default_factory=list)
    result: str = "1/2-1/2"
    termination: str = "unknown"
    plies: int = 0
    moves_uci: List[str] = field(default_factory=list)
    resigned: bool = False
    duration_seconds: float = 0.0
    mean_root_value: float = 0.0
    mean_policy_entropy: float = 0.0
    mean_value_gap: float = 0.0
    """Mean |z - q|: how much the search disagreed with the eventual result.
    High and falling is the signature of a value head that is learning."""


@dataclass
class SelfPlayBatch:
    games: List[GameRecord] = field(default_factory=list)
    duration_seconds: float = 0.0
    inference: Optional[Dict[str, float]] = None
    """Shared-GPU server statistics, when one was used."""

    @property
    def positions(self) -> int:
        return sum(len(g.samples) for g in self.games)

    def summary(self) -> Dict[str, float]:
        if not self.games:
            return {}
        results = [g.result for g in self.games]
        plies = [g.plies for g in self.games]
        return {
            "games": len(self.games),
            "positions": self.positions,
            "white_wins": results.count("1-0"),
            "black_wins": results.count("0-1"),
            "draws": results.count("1/2-1/2"),
            "draw_rate": results.count("1/2-1/2") / len(results),
            "avg_plies": float(np.mean(plies)),
            "max_plies": int(np.max(plies)),
            "resignations": sum(1 for g in self.games if g.resigned),
            "avg_root_value": float(np.mean([g.mean_root_value for g in self.games])),
            "avg_policy_entropy": float(
                np.mean([g.mean_policy_entropy for g in self.games])
            ),
            "avg_value_gap": float(np.mean([g.mean_value_gap for g in self.games])),
            "duration_seconds": self.duration_seconds,
            "games_per_minute": (
                len(self.games) / (self.duration_seconds / 60.0)
                if self.duration_seconds > 0
                else 0.0
            ),
        }


# --------------------------------------------------------------------------- #
# Single game
# --------------------------------------------------------------------------- #


def play_game(
    evaluator: BaseEvaluator,
    config: RunConfig,
    rng: np.random.Generator,
    *,
    allow_resign: bool = True,
    watch: Optional[WatchPublisher] = None,
    iteration: int = 0,
    game_index: int = 0,
) -> GameRecord:
    """Play one complete self-play game and return its training records."""
    started = time.perf_counter()
    history_length = config.network.history_length
    state = PositionHistory(chess.Board(), history_length=history_length)
    searcher = MCTS(evaluator, config.search, rng)

    for _ in range(config.selfplay.opening_random_plies):
        legal = list(state.board.legal_moves)
        if not legal:
            break
        state.push(legal[int(rng.integers(len(legal)))])

    def publish(move: chess.Move, search) -> None:
        """Show the position just reached, then hold it if the observer asked."""
        if watch is None or not watch.enabled:
            return
        watch.record(
            state.board,
            last_move=move.uci(),
            value=search.root_value,
            top_moves=[(option.uci(), prior) for option, prior, _ in search.top_moves(4)],
        )
        watch.pace()

    if watch is not None:
        watch.begin(state.board, iteration=iteration, game_index=game_index)

    planes_log: List[np.ndarray] = []
    policy_log: List[Dict[int, float]] = []
    turn_log: List[bool] = []
    root_values: List[float] = []
    entropies: List[float] = []
    moves_uci: List[str] = []

    record = GameRecord()
    resign_streak = {chess.WHITE: 0, chess.BLACK: 0}
    termination_reason = "max_plies"
    result: Optional[str] = None

    while state.ply < config.selfplay.max_game_plies:
        termination = terminal_state(state)
        if termination is not None:
            termination_reason = termination.reason
            result = result_string(state, termination)
            break

        planes = state.encode()
        temperature = (
            config.search.temperature
            if state.ply < config.search.temperature_moves
            else 0.0
        )
        move, search_result = searcher.select_move(
            state, add_noise=True, temperature=temperature
        )
        if move is None or search_result.best_move is None:
            termination_reason = "no_legal_moves"
            result = "1/2-1/2"
            break

        planes_log.append(planes)
        policy_log.append(dict(search_result.policy_target))
        turn_log.append(state.board.turn)
        root_values.append(search_result.root_value)
        entropies.append(_entropy(search_result.policy_target))

        mover = state.board.turn
        if allow_resign and config.selfplay.resign_threshold is not None:
            if search_result.root_value < config.selfplay.resign_threshold:
                resign_streak[mover] += 1
            else:
                resign_streak[mover] = 0
            if resign_streak[mover] >= config.selfplay.resign_consecutive:
                result = "0-1" if mover == chess.WHITE else "1-0"
                termination_reason = "resignation"
                record.resigned = True
                moves_uci.append(move.uci())
                state.push(move)
                publish(move, search_result)
                break

        moves_uci.append(move.uci())
        state.push(move)
        publish(move, search_result)
    else:
        termination = terminal_state(state)
        if termination is not None:
            termination_reason = termination.reason
            result = result_string(state, termination)

    if result is None:
        result = "1/2-1/2"

    white_score = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}[result]
    outcomes = [
        white_score if turn == chess.WHITE else -white_score for turn in turn_log
    ]
    targets = _value_targets(outcomes, root_values, config)
    record.samples = [
        make_sample(planes, policy, target, history_length)
        for planes, policy, target in zip(planes_log, policy_log, targets)
    ]
    record.result = result
    record.termination = termination_reason
    record.plies = state.ply
    record.moves_uci = moves_uci
    record.duration_seconds = time.perf_counter() - started
    record.mean_root_value = float(np.mean(root_values)) if root_values else 0.0
    record.mean_policy_entropy = float(np.mean(entropies)) if entropies else 0.0
    record.mean_value_gap = (
        float(np.mean(np.abs(np.asarray(outcomes) - np.asarray(root_values))))
        if root_values
        else 0.0
    )

    if watch is not None:
        watch.end(
            state.board,
            result=result,
            termination=termination_reason,
            resigned=record.resigned,
        )
    return record


def _value_targets(
    outcomes: List[float], root_values: List[float], config: RunConfig
) -> List[float]:
    """Blend the game result with what the search thought of each position.

    ``z`` alone is a single Bernoulli-ish draw stamped onto every position of the
    game: unbiased, but so noisy and so correlated that the value head spends its
    early capacity fitting the outcome of *games* rather than the merit of
    *positions*. ``q`` -- the root value the tree settled on -- is a much
    lower-variance opinion about this position specifically, and it comes from the
    network's own search, so nothing external enters the target.

    With ``simulations == 1`` there is no tree: ``q`` is the value head's raw
    output and mixing it in would regress the head onto itself, so the weight is
    ignored.
    """
    weight = float(config.train.value_search_weight)
    if weight <= 0.0 or config.search.simulations <= 1:
        return outcomes
    weight = min(weight, 1.0)
    return [
        (1.0 - weight) * outcome + weight * value
        for outcome, value in zip(outcomes, root_values)
    ]


def _entropy(distribution: Dict[int, float]) -> float:
    values = np.fromiter(distribution.values(), dtype=np.float64, count=len(distribution))
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    return float(-(values * np.log(values)).sum())


def game_to_pgn(record: GameRecord, white: str = "NN", black: str = "NN") -> str:
    game = chess.pgn.Game()
    game.headers["White"] = white
    game.headers["Black"] = black
    game.headers["Result"] = record.result
    game.headers["Termination"] = record.termination
    node = game
    for uci in record.moves_uci:
        node = node.add_variation(chess.Move.from_uci(uci))
    return str(game)


# --------------------------------------------------------------------------- #
# Worker plumbing
# --------------------------------------------------------------------------- #

_WORKER_CACHE: Dict[Tuple[str, float], object] = {}


def _worker_evaluator(checkpoint: str, device: str, threads: int) -> Evaluator:
    import torch

    torch.set_num_threads(max(1, threads))
    key = (checkpoint, os.path.getmtime(checkpoint))
    cached = _WORKER_CACHE.get(key)
    if cached is None:
        _WORKER_CACHE.clear()
        model, _ = load_checkpoint(checkpoint, device=device)
        cached = Evaluator(model, device=device)
        _WORKER_CACHE[key] = cached
    return cached  # type: ignore[return-value]


def _worker_play(payload: Dict[str, Any]) -> List[GameRecord]:
    config = RunConfig.from_dict(payload["config"])
    # A worker that has been handed a line to the shared GPU owns no model at
    # all -- which also saves deserialising the checkpoint once per worker per
    # iteration.
    evaluator: BaseEvaluator = worker_evaluator() or _worker_evaluator(
        payload["checkpoint"],
        config.selfplay.device,
        config.selfplay.torch_threads_per_worker,
    )
    rng = np.random.default_rng(payload["seed"])
    iteration = payload.get("iteration", 0)
    watch = (
        WatchPublisher(
            Path(payload["watch_dir"]),
            payload["slot"],
            run_id=payload.get("run_id"),
        )
        if payload.get("watch_dir")
        else None
    )
    return [
        play_game(
            evaluator,
            config,
            rng,
            allow_resign=payload["allow_resign"],
            watch=watch,
            iteration=iteration,
            game_index=index,
        )
        for index in range(payload["count"])
    ]


def generate_selfplay(
    checkpoint: Path,
    config: RunConfig,
    *,
    games: Optional[int] = None,
    seed: int = 0,
    progress=None,
    watch_dir: Optional[Path] = None,
    iteration: int = 0,
    run_id: Optional[str] = None,
) -> SelfPlayBatch:
    """Generate ``games`` self-play games using a pool of worker processes.

    ``watch_dir`` turns each worker into a publisher for the dashboard: it
    writes the game it is playing into its own slot file there. Whether anything
    is actually written is decided by the settings file in that directory, which
    the workers poll, so watching can be toggled mid-run.
    """
    total_games = games if games is not None else config.selfplay.games_per_iteration
    workers = max(1, min(config.selfplay.workers, total_games))
    started = time.perf_counter()
    batch = SelfPlayBatch()

    chunks = _split_evenly(total_games, workers)
    no_resign_target = int(round(total_games * config.selfplay.resign_disable_fraction))

    payloads = []
    assigned_no_resign = 0
    for index, count in enumerate(chunks):
        allow_resign = True
        if assigned_no_resign < no_resign_target:
            allow_resign = False
            assigned_no_resign += count
        payloads.append(
            {
                "checkpoint": str(checkpoint),
                "config": config.to_dict(),
                "seed": seed * 100003 + index,
                "count": count,
                "allow_resign": allow_resign,
                "slot": index,
                "watch_dir": str(watch_dir) if watch_dir is not None else None,
                "iteration": iteration,
                "run_id": run_id,
            }
        )

    if workers == 1:
        for payload in payloads:
            batch.games.extend(_worker_play(payload))
            if progress:
                progress(len(batch.games), total_games)
    else:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        server = _start_inference_server(checkpoint, config, workers, context)
        initializer = None
        initargs: Tuple = ()
        if server is not None:
            initializer = worker_initializer
            initargs = (server.requests, server.responses, server.counter)
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=initializer,
                initargs=initargs,
            ) as pool:
                futures = [pool.submit(_worker_play, payload) for payload in payloads]
                for future in as_completed(futures):
                    batch.games.extend(future.result())
                    if progress:
                        progress(len(batch.games), total_games)
        finally:
            if server is not None:
                batch.inference = server.stop().to_dict()

    batch.duration_seconds = time.perf_counter() - started
    return batch


def _start_inference_server(
    checkpoint: Path, config: RunConfig, workers: int, context
) -> Optional[InferenceServer]:
    """Bring up the shared GPU, or return ``None`` and let workers use their own.

    Falling back is deliberate: a machine without a GPU, or a checkpoint that
    will not load, must still be able to generate self-play -- just slower.
    """
    if not config.inference.enabled:
        return None
    device = resolve_device(config.inference.device)
    if device == "cpu":
        # Sharing one CPU model between workers would serialise them behind a
        # single core; the per-worker path is strictly better there.
        return None
    try:
        server = InferenceServer(
            checkpoint,
            device=device,
            max_batch=config.inference.max_batch,
            collect_timeout_ms=config.inference.collect_timeout_ms,
            use_amp=config.inference.use_amp,
            request_queue=context.Queue(),
            response_queues=[context.Queue() for _ in range(workers)],
            counter=context.Value("i", 0),
        )
        return server.start()
    except Exception:  # noqa: BLE001 - never let inference kill a training run
        return None


def _split_evenly(total: int, parts: int) -> List[int]:
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]
