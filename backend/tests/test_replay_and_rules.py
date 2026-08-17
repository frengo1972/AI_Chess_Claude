"""Replay-buffer compression and the termination rules used by self-play."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from app.engine.encoding import PositionHistory, legal_move_indices
from app.engine.replay import (
    ReplayBuffer,
    make_sample,
    pack_planes,
    unpack_planes,
)
from app.engine.rules import DRAW_HALFMOVE_CLOCK, terminal_state


@pytest.mark.parametrize("history", [1, 4, 8])
def test_pack_unpack_is_lossless(history):
    state = PositionHistory(chess.Board(), history_length=history)
    for uci in ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4"]:
        state.push(chess.Move.from_uci(uci))
    planes = state.encode()
    bits, scalars = pack_planes(planes, history)
    restored = unpack_planes(bits, scalars, history)
    assert np.array_equal(planes, restored)


def test_packing_is_much_smaller_than_raw_planes():
    state = PositionHistory(chess.Board(), history_length=8)
    planes = state.encode()
    bits, scalars = pack_planes(planes, 8)
    raw = planes.nbytes
    packed = bits.nbytes + scalars.nbytes
    assert packed * 20 < raw


def test_buffer_sampling_shapes_and_normalisation():
    history = 2
    buffer = ReplayBuffer(capacity=64, history_length=history)
    state = PositionHistory(chess.Board(), history_length=history)
    for _ in range(8):
        policy = {index: 1.0 for index in legal_move_indices(state.board)}
        buffer.extend([make_sample(state.encode(), policy, 0.5, history)])
        state.push(next(iter(state.board.legal_moves)))

    planes, policies, values = buffer.sample_batch(4, np.random.default_rng(0))
    assert planes.shape == (4, 14 * history + 7, 8, 8)
    assert policies.shape[0] == 4
    assert np.allclose(policies.sum(axis=1), 1.0, atol=1e-3)
    assert values.shape == (4,)


def test_buffer_respects_capacity():
    buffer = ReplayBuffer(capacity=5, history_length=1)
    state = PositionHistory(chess.Board(), history_length=1)
    policy = {index: 1.0 for index in legal_move_indices(state.board)}
    for _ in range(20):
        buffer.extend([make_sample(state.encode(), policy, 0.0, 1)])
    assert len(buffer) == 5
    assert buffer.total_added == 20


def test_buffer_roundtrips_through_disk(tmp_path):
    history = 2
    buffer = ReplayBuffer(capacity=32, history_length=history)
    state = PositionHistory(chess.Board(), history_length=history)
    policy = {index: 1.0 for index in legal_move_indices(state.board)}
    buffer.extend([make_sample(state.encode(), policy, -1.0, history) for _ in range(6)])

    path = tmp_path / "shard.npz"
    buffer.save(path)
    restored = ReplayBuffer(capacity=32, history_length=history)
    assert restored.load(path) == 6

    original = buffer.sample_batch(6, np.random.default_rng(1))
    reloaded = restored.sample_batch(6, np.random.default_rng(1))
    assert np.array_equal(original[0], reloaded[0])
    assert np.allclose(original[2], reloaded[2])


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #


def test_checkmate_is_a_loss_for_the_side_to_move():
    state = PositionHistory(chess.Board("7k/5QQ1/8/8/8/8/8/K7 b - - 0 1"))
    termination = terminal_state(state)
    assert termination is not None
    assert termination.reason == "checkmate"
    assert termination.value == -1.0


def test_stalemate_and_insufficient_material_are_draws():
    stalemate = PositionHistory(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"))
    assert terminal_state(stalemate).reason == "stalemate"

    bare = PositionHistory(chess.Board("7k/8/6K1/8/8/8/8/8 w - - 0 1"))
    assert terminal_state(bare).reason == "insufficient_material"


def test_threefold_repetition_is_automatic_for_selfplay():
    state = PositionHistory(chess.Board())
    assert terminal_state(state) is None
    for _ in range(2):
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
            state.push(chess.Move.from_uci(uci))
    termination = terminal_state(state)
    assert termination is not None
    assert termination.reason == "threefold_repetition"
    assert termination.value == 0.0


def test_fifty_move_rule_is_automatic():
    board = chess.Board("8/8/4k3/8/8/4K3/8/7R w - - 0 1")
    board.halfmove_clock = DRAW_HALFMOVE_CLOCK
    state = PositionHistory(board)
    assert terminal_state(state).reason == "fifty_move_rule"
