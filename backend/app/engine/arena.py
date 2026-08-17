"""Head-to-head matches: gating a new network, and estimating Elo.

Two uses:

* **Gating** -- a freshly trained candidate only becomes the new "best"
  network if it scores above ``ArenaConfig.win_threshold`` against the current
  best. This is what keeps training monotone; without it a bad iteration can
  poison the replay buffer for a long time.
* **Measurement** -- the same code produces the score that the Elo curve on the
  training dashboard is built from.

Games are deterministic apart from a few random opening plies, which is what
keeps a match from being the same game N times.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from math import log10
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np

from app.config import RunConfig
from app.engine.encoding import PositionHistory
from app.engine.evaluator import Evaluator
from app.engine.mcts import MCTS
from app.engine.network import load_checkpoint
from app.engine.rules import result_string, terminal_state

DEFAULT_OPENING_PLIES = 4


@dataclass
class MatchResult:
    challenger_wins: int = 0
    champion_wins: int = 0
    draws: int = 0
    games: List[Dict] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def total(self) -> int:
        return self.challenger_wins + self.champion_wins + self.draws

    @property
    def score(self) -> float:
        """Challenger's score in ``[0, 1]`` (win = 1, draw = 0.5)."""
        if self.total == 0:
            return 0.5
        return (self.challenger_wins + 0.5 * self.draws) / self.total

    @property
    def elo_difference(self) -> float:
        return score_to_elo(self.score)

    def summary(self) -> Dict[str, float]:
        return {
            "games": self.total,
            "challenger_wins": self.challenger_wins,
            "champion_wins": self.champion_wins,
            "draws": self.draws,
            "score": self.score,
            "elo_difference": self.elo_difference,
            "elo_error_95": elo_error_margin(self.score, self.total),
            "duration_seconds": self.duration_seconds,
        }


def score_to_elo(score: float, cap: float = 800.0) -> float:
    """Convert a match score into an Elo difference, clamped for degenerate scores."""
    epsilon = 1e-4
    clamped = min(max(score, epsilon), 1 - epsilon)
    return max(-cap, min(cap, -400.0 * log10(1.0 / clamped - 1.0)))


def elo_error_margin(score: float, games: int) -> float:
    """Rough 95% confidence half-width on the Elo difference."""
    if games <= 1:
        return float("inf")
    variance = max(score * (1.0 - score), 1e-6)
    standard_error = (variance / games) ** 0.5
    low = score_to_elo(max(score - 1.96 * standard_error, 1e-4))
    high = score_to_elo(min(score + 1.96 * standard_error, 1 - 1e-4))
    return (high - low) / 2.0


# --------------------------------------------------------------------------- #
# Playing a single game between two searchers
# --------------------------------------------------------------------------- #


def play_match_game(
    white_searcher: MCTS,
    black_searcher: MCTS,
    config: RunConfig,
    rng: np.random.Generator,
    *,
    simulations: Optional[int] = None,
    max_plies: int = 240,
    opening_plies: int = DEFAULT_OPENING_PLIES,
) -> Tuple[str, str, List[str]]:
    """Return ``(result, termination_reason, moves_uci)``."""
    state = PositionHistory(
        chess.Board(), history_length=config.network.history_length
    )
    moves: List[str] = []

    for _ in range(opening_plies):
        legal = list(state.board.legal_moves)
        if not legal:
            break
        move = legal[int(rng.integers(len(legal)))]
        moves.append(move.uci())
        state.push(move)

    while state.ply < max_plies:
        termination = terminal_state(state)
        if termination is not None:
            return result_string(state, termination), termination.reason, moves

        searcher = white_searcher if state.board.turn == chess.WHITE else black_searcher
        move, _ = searcher.select_move(
            state, add_noise=False, temperature=0.0, simulations=simulations
        )
        if move is None:
            break
        moves.append(move.uci())
        state.push(move)

    termination = terminal_state(state)
    if termination is not None:
        return result_string(state, termination), termination.reason, moves
    return "1/2-1/2", "max_plies", moves


# --------------------------------------------------------------------------- #
# Worker plumbing
# --------------------------------------------------------------------------- #

_ARENA_CACHE: Dict[str, Evaluator] = {}


def _arena_evaluator(checkpoint: str, device: str, threads: int) -> Evaluator:
    import torch

    torch.set_num_threads(max(1, threads))
    cached = _ARENA_CACHE.get(checkpoint)
    if cached is None:
        model, _ = load_checkpoint(checkpoint, device=device)
        cached = Evaluator(model, device=device)
        _ARENA_CACHE[checkpoint] = cached
    return cached


def _arena_worker(payload: Tuple[str, str, dict, int, List[bool], int, int]) -> List[Dict]:
    challenger, champion, config_dict, seed, challenger_is_white, simulations, max_plies = payload
    config = RunConfig.from_dict(config_dict)
    device = config.selfplay.device
    threads = config.selfplay.torch_threads_per_worker

    challenger_eval = _arena_evaluator(challenger, device, threads)
    champion_eval = _arena_evaluator(champion, device, threads)
    rng = np.random.default_rng(seed)

    challenger_mcts = MCTS(challenger_eval, config.search, rng)
    champion_mcts = MCTS(champion_eval, config.search, rng)

    outcomes: List[Dict] = []
    for index, white_is_challenger in enumerate(challenger_is_white):
        white = challenger_mcts if white_is_challenger else champion_mcts
        black = champion_mcts if white_is_challenger else challenger_mcts
        result, reason, moves = play_match_game(
            white,
            black,
            config,
            rng,
            simulations=simulations,
            max_plies=max_plies,
        )
        outcomes.append(
            {
                "result": result,
                "termination": reason,
                "plies": len(moves),
                "challenger_white": white_is_challenger,
                "moves": moves,
            }
        )
    return outcomes


def run_match(
    challenger: Path,
    champion: Path,
    config: RunConfig,
    *,
    games: Optional[int] = None,
    simulations: Optional[int] = None,
    max_plies: Optional[int] = None,
    seed: int = 0,
    workers: Optional[int] = None,
) -> MatchResult:
    """Play a gating match; colours alternate so both nets get equal Whites."""
    total = games if games is not None else config.arena.games
    sims = simulations if simulations is not None else config.arena.simulations
    plies = max_plies if max_plies is not None else config.arena.max_plies
    pool_size = max(1, min(workers or config.selfplay.workers, total))

    colours = [i % 2 == 0 for i in range(total)]
    chunks: List[List[bool]] = [[] for _ in range(pool_size)]
    for index, colour in enumerate(colours):
        chunks[index % pool_size].append(colour)

    payloads = [
        (
            str(challenger),
            str(champion),
            config.to_dict(),
            seed * 7919 + index,
            chunk,
            sims,
            plies,
        )
        for index, chunk in enumerate(chunks)
        if chunk
    ]

    started = time.perf_counter()
    outcomes: List[Dict] = []
    if pool_size == 1:
        for payload in payloads:
            outcomes.extend(_arena_worker(payload))
    else:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=pool_size, mp_context=context) as pool:
            futures = [pool.submit(_arena_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                outcomes.extend(future.result())

    match = MatchResult(duration_seconds=time.perf_counter() - started)
    for outcome in outcomes:
        result = outcome["result"]
        if result == "1/2-1/2":
            match.draws += 1
        else:
            white_won = result == "1-0"
            challenger_won = white_won == outcome["challenger_white"]
            if challenger_won:
                match.challenger_wins += 1
            else:
                match.champion_wins += 1
    match.games = outcomes
    return match
