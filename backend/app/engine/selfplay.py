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
from app.engine.evaluator import Evaluator
from app.engine.mcts import MCTS
from app.engine.network import load_checkpoint
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


@dataclass
class SelfPlayBatch:
    games: List[GameRecord] = field(default_factory=list)
    duration_seconds: float = 0.0

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
    evaluator: Evaluator,
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
    record.samples = [
        make_sample(
            planes,
            policy,
            white_score if turn == chess.WHITE else -white_score,
            history_length,
        )
        for planes, policy, turn in zip(planes_log, policy_log, turn_log)
    ]
    record.result = result
    record.termination = termination_reason
    record.plies = state.ply
    record.moves_uci = moves_uci
    record.duration_seconds = time.perf_counter() - started
    record.mean_root_value = float(np.mean(root_values)) if root_values else 0.0
    record.mean_policy_entropy = float(np.mean(entropies)) if entropies else 0.0

    if watch is not None:
        watch.end(
            state.board,
            result=result,
            termination=termination_reason,
            resigned=record.resigned,
        )
    return record


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
    evaluator = _worker_evaluator(
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
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(_worker_play, payload) for payload in payloads]
            for future in as_completed(futures):
                batch.games.extend(future.result())
                if progress:
                    progress(len(batch.games), total_games)

    batch.duration_seconds = time.perf_counter() - started
    return batch


def _split_evenly(total: int, parts: int) -> List[int]:
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]
