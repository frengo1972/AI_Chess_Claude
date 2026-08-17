"""Search behaviour that must hold even with a randomly initialised network."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from app.config import NetworkConfig, SearchConfig
from app.engine.encoding import PositionHistory
from app.engine.evaluator import Evaluator
from app.engine.mcts import MCTS
from app.engine.network import ChessNet
from app.engine.rules import terminal_state

HISTORY = 2


@pytest.fixture(scope="module")
def searcher_factory():
    model = ChessNet(NetworkConfig(history_length=HISTORY, residual_blocks=2, filters=32))
    evaluator = Evaluator(model, "cpu")

    def build(**overrides):
        config = SearchConfig(**{"simulations": 128, "max_batch_size": 8, **overrides})
        return MCTS(evaluator, config, np.random.default_rng(1234))

    return build


def _state(fen: str) -> PositionHistory:
    return PositionHistory(chess.Board(fen), history_length=HISTORY)


def test_finds_mate_in_one(searcher_factory):
    """Terminal values back up exactly, so a mate must win the visit count."""
    state = _state("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    searcher = searcher_factory(simulations=400)
    result = searcher.search(state)
    assert result.best_move == chess.Move.from_uci("a1a8")
    assert result.root_value > 0.5


def test_avoids_being_mated_when_only_one_move_helps(searcher_factory):
    state = _state("7k/5ppp/8/8/8/8/8/R5K1 b - - 0 1")
    searcher = searcher_factory(simulations=256)
    result = searcher.search(state)
    assert result.best_move is not None
    assert result.best_move in state.board.legal_moves


def test_single_simulation_is_pure_policy(searcher_factory):
    state = _state(chess.STARTING_FEN)
    searcher = searcher_factory(simulations=1)
    result = searcher.search(state)
    assert result.simulations == 1
    assert result.used_search is False
    assert result.best_move in state.board.legal_moves
    assert pytest.approx(sum(result.policy_target.values()), rel=1e-4) == 1.0
    assert len(result.policy_target) == state.board.legal_moves.count()


def test_search_never_mutates_the_state(searcher_factory):
    state = _state("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    fen_before = state.board.fen()
    planes_before = state.encode()
    searcher = searcher_factory(simulations=200)
    searcher.search(state, add_noise=True)
    assert state.board.fen() == fen_before
    assert np.array_equal(state.encode(), planes_before)
    assert state.ply == 0


def test_policy_only_covers_all_legal_moves_and_nothing_else(searcher_factory):
    state = _state("8/8/8/3k4/8/3K4/8/7R w - - 0 1")
    searcher = searcher_factory(simulations=64)
    result = searcher.search(state)
    legal = set(state.board.legal_moves)
    assert set(result.move_by_index.values()) == legal


def test_terminal_position_returns_no_move(searcher_factory):
    state = _state("7k/5QQ1/8/8/8/8/8/K7 b - - 0 1")
    assert terminal_state(state) is not None
    searcher = searcher_factory(simulations=16)
    result = searcher.search(state)
    assert result.best_move is None
    assert result.terminal is not None
    assert result.terminal.reason == "checkmate"


def test_visit_counts_sum_to_simulation_budget(searcher_factory):
    state = _state(chess.STARTING_FEN)
    searcher = searcher_factory(simulations=100)
    result = searcher.search(state)
    # The root evaluation itself consumes one simulation.
    assert sum(result.visit_counts.values()) == result.simulations - 1


def test_temperature_zero_is_deterministic(searcher_factory):
    state = _state(chess.STARTING_FEN)
    first = searcher_factory(simulations=120).select_move(state, temperature=0.0)[0]
    second = searcher_factory(simulations=120).select_move(state, temperature=0.0)[0]
    assert first == second
