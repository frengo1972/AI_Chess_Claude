"""The encoding is the part where a silent bug quietly ruins a whole training run."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from app.engine.encoding import (
    POLICY_SIZE,
    PositionHistory,
    index_to_move,
    legal_move_indices,
    legal_move_mask,
    move_to_index,
    num_input_planes,
)


def _random_positions(count: int = 60, seed: int = 7):
    rng = np.random.default_rng(seed)
    boards = []
    board = chess.Board()
    while len(boards) < count:
        if board.is_game_over(claim_draw=False):
            board = chess.Board()
            continue
        boards.append(board.copy())
        moves = list(board.legal_moves)
        board.push(moves[int(rng.integers(len(moves)))])
    return boards


def test_move_index_roundtrip_over_random_games():
    for board in _random_positions():
        white_to_move = board.turn == chess.WHITE
        for move in board.legal_moves:
            index = move_to_index(move, white_to_move)
            assert 0 <= index < POLICY_SIZE
            assert index_to_move(index, board) == move


def test_indices_are_unique_per_position():
    for board in _random_positions():
        mapping = legal_move_indices(board)
        assert len(mapping) == board.legal_moves.count()


def test_underpromotions_get_distinct_indices():
    board = chess.Board("8/4P3/8/8/8/8/8/K6k w - - 0 1")
    mapping = legal_move_indices(board)
    promotions = {
        move.promotion for move in mapping.values() if move.promotion is not None
    }
    assert promotions == {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT}
    assert len({move_to_index(m, True) for m in mapping.values()}) == len(mapping)


def test_black_promotion_roundtrip():
    board = chess.Board("K6k/8/8/8/8/8/4p3/8 b - - 0 1")
    for move in board.legal_moves:
        assert index_to_move(move_to_index(move, False), board) == move


def test_castling_roundtrip_both_colours():
    white = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    black = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    for board in (white, black):
        castles = [m for m in board.legal_moves if board.is_castling(m)]
        assert len(castles) == 2
        for move in castles:
            index = move_to_index(move, board.turn == chess.WHITE)
            assert index_to_move(index, board) == move


def test_en_passant_roundtrip():
    board = chess.Board("8/8/8/3pP3/8/8/8/K6k w - d6 0 1")
    ep_moves = [m for m in board.legal_moves if board.is_en_passant(m)]
    assert ep_moves
    for move in ep_moves:
        assert index_to_move(move_to_index(move, True), board) == move


def test_plane_shape_and_history_padding():
    for history in (1, 2, 4, 8):
        state = PositionHistory(chess.Board(), history_length=history)
        planes = state.encode()
        assert planes.shape == (num_input_planes(history), 8, 8)
        assert planes.dtype == np.float32
        # Only the newest step is populated at the start of a game.
        assert planes[0:12].sum() == 32


def test_orientation_makes_mirrored_positions_identical():
    """A position and its colour-swapped mirror must encode identically."""
    white = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    black = white.mirror()
    encoded_white = PositionHistory(white, history_length=1).encode()
    encoded_black = PositionHistory(black, history_length=1).encode()
    # Plane 0 of the constants block is the absolute colour, which differs by design.
    piece_planes = slice(0, 14)
    assert np.array_equal(encoded_white[piece_planes], encoded_black[piece_planes])


def test_repetition_planes_light_up():
    state = PositionHistory(chess.Board(), history_length=2)
    shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    for uci in shuffle:
        state.push(chess.Move.from_uci(uci))
    assert state.repetition_count() == 2
    planes = state.encode()
    assert planes[12].max() == 1.0  # "seen before" flag on the newest step


def test_push_pop_restores_state():
    state = PositionHistory(chess.Board(), history_length=4)
    before = state.encode()
    move = next(iter(state.board.legal_moves))
    state.push(move)
    popped = state.pop()
    assert popped == move
    assert np.array_equal(state.encode(), before)
    assert state.ply == 0


def test_legal_mask_matches_move_generator():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    mask = legal_move_mask(board)
    assert mask.sum() == board.legal_moves.count()
    for index in np.flatnonzero(mask):
        assert index_to_move(int(index), board) in board.legal_moves


@pytest.mark.parametrize("history", [1, 3, 8])
def test_history_window_slides(history):
    state = PositionHistory(chess.Board(), history_length=history)
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]:
        state.push(chess.Move.from_uci(uci))
    planes = state.encode()
    assert planes.shape[0] == num_input_planes(history)
    assert np.isfinite(planes).all()
