"""Measuring the network against Stockfish at a capped strength.

This is the *only* place where the two engines meet during training, and the
direction of information flow is strictly one-way: Stockfish produces a score,
the score becomes a KPI. No Stockfish move, evaluation or line is ever written
to the replay buffer, and ``app.engine`` never imports this module at import
time (``train.py`` pulls it in lazily inside a ``try``).

The Elo estimate is anchored: if the network scores ``s`` against an opponent
of known strength ``E``, then ``elo(net) ~ E + 400*log10(s/(1-s))``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import chess
import numpy as np

from app.config import BenchmarkConfig, RunConfig, SearchConfig
from app.engine.arena import score_to_elo, elo_error_margin
from app.engine.encoding import PositionHistory
from app.engine.evaluator import Evaluator
from app.engine.mcts import MCTS
from app.engine.network import load_checkpoint
from app.engine.rules import result_string, terminal_state
from app.services.stockfish_service import EngineLevel, StockfishService, get_stockfish


def play_network_vs_stockfish(
    checkpoint: Path,
    config: RunConfig,
    *,
    games: int = 10,
    stockfish_elo: int = 1350,
    movetime_ms: int = 50,
    simulations: Optional[int] = None,
    max_plies: int = 240,
    opening_plies: int = 4,
    seed: int = 0,
    device: str = "cpu",
    stockfish: Optional[StockfishService] = None,
    progress=None,
) -> Dict:
    """Play ``games`` alternating-colour games and return the score summary."""
    engine = stockfish or get_stockfish()
    if not engine.available:
        raise RuntimeError("Stockfish is not available for benchmarking")

    model, _ = load_checkpoint(checkpoint, device=device)
    evaluator = Evaluator(model, device)
    search_config = SearchConfig(
        **{**vars(config.search), "dirichlet_epsilon": 0.0}
    )
    rng = np.random.default_rng(seed)
    searcher = MCTS(evaluator, search_config, rng)
    level = EngineLevel(elo=stockfish_elo, movetime_ms=movetime_ms, depth=None)

    wins = draws = losses = 0
    played: List[Dict] = []
    started = time.perf_counter()

    for index in range(games):
        network_is_white = index % 2 == 0
        state = PositionHistory(
            chess.Board(), history_length=config.network.history_length
        )
        for _ in range(opening_plies):
            legal = list(state.board.legal_moves)
            if not legal:
                break
            state.push(legal[int(rng.integers(len(legal)))])

        result = "1/2-1/2"
        reason = "max_plies"
        while state.ply < max_plies:
            termination = terminal_state(state)
            if termination is not None:
                result = result_string(state, termination)
                reason = termination.reason
                break

            network_turn = (state.board.turn == chess.WHITE) == network_is_white
            if network_turn:
                move, _ = searcher.select_move(
                    state, add_noise=False, temperature=0.0, simulations=simulations
                )
            else:
                move = engine.best_move(state.board.fen(), level)
            if move is None:
                break
            state.push(move)
        else:
            termination = terminal_state(state)
            if termination is not None:
                result = result_string(state, termination)
                reason = termination.reason

        if result == "1/2-1/2":
            draws += 1
            outcome = "draw"
        else:
            network_won = (result == "1-0") == network_is_white
            if network_won:
                wins += 1
                outcome = "win"
            else:
                losses += 1
                outcome = "loss"

        played.append(
            {
                "index": index,
                "network_white": network_is_white,
                "result": result,
                "outcome": outcome,
                "termination": reason,
                "plies": state.ply,
            }
        )
        if progress:
            progress(index + 1, games)

    total = max(wins + draws + losses, 1)
    score = (wins + 0.5 * draws) / total
    return {
        "games": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "opponent_elo": stockfish_elo,
        "estimated_elo": stockfish_elo + score_to_elo(score),
        "elo_error_95": elo_error_margin(score, total),
        "duration_seconds": time.perf_counter() - started,
        "detail": played,
    }


def benchmark_against_stockfish(
    checkpoint: Path, config: RunConfig, benchmark: BenchmarkConfig
) -> Dict[str, float]:
    """Trainer-facing wrapper that returns only the KPI columns."""
    summary = play_network_vs_stockfish(
        checkpoint,
        config,
        games=benchmark.games,
        stockfish_elo=benchmark.stockfish_elo,
        movetime_ms=benchmark.movetime_ms,
        simulations=benchmark.simulations,
    )
    return {
        "benchmark_elo": summary["estimated_elo"],
        "benchmark_score": summary["score"],
        "benchmark_games": summary["games"],
        "benchmark_opponent_elo": summary["opponent_elo"],
    }
