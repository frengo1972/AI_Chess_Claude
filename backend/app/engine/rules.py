"""Game-termination rules, expressed the way the search and the trainer need them.

``python-chess`` already implements the full rules of chess; this module only
decides *which* draw conditions the engine treats as terminal. Repetition and
the 50-move rule are claimable rather than automatic in FIDE rules, but for
self-play they must be automatic, otherwise the network happily shuffles
forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chess

from app.engine.encoding import PositionHistory

DRAW_REPETITION_COUNT = 3
DRAW_HALFMOVE_CLOCK = 100  # 50 full moves without a pawn move or capture


@dataclass(frozen=True)
class Termination:
    """Why a game ended, and the value for the side to move at that node."""

    reason: str
    value: float
    """+1 win / 0 draw / -1 loss, from the point of view of the side to move."""

    @property
    def is_draw(self) -> bool:
        return self.value == 0.0


def terminal_state(state: PositionHistory) -> Optional[Termination]:
    """Return the termination of ``state``, or ``None`` if the game continues."""
    board = state.board

    if board.is_checkmate():
        # The side to move has been mated: from its own perspective it lost.
        return Termination("checkmate", -1.0)
    if board.is_stalemate():
        return Termination("stalemate", 0.0)
    if board.is_insufficient_material():
        return Termination("insufficient_material", 0.0)
    if board.halfmove_clock >= DRAW_HALFMOVE_CLOCK:
        return Termination("fifty_move_rule", 0.0)
    if state.repetition_count() >= DRAW_REPETITION_COUNT:
        return Termination("threefold_repetition", 0.0)
    return None


def terminal_state_from_board(board: chess.Board) -> Optional[Termination]:
    """Same as :func:`terminal_state` for a bare board (slower repetition test)."""
    if board.is_checkmate():
        return Termination("checkmate", -1.0)
    if board.is_stalemate():
        return Termination("stalemate", 0.0)
    if board.is_insufficient_material():
        return Termination("insufficient_material", 0.0)
    if board.halfmove_clock >= DRAW_HALFMOVE_CLOCK:
        return Termination("fifty_move_rule", 0.0)
    if board.is_repetition(DRAW_REPETITION_COUNT):
        return Termination("threefold_repetition", 0.0)
    return None


def result_string(state: PositionHistory, termination: Termination) -> str:
    """PGN-style result for a finished game."""
    if termination.is_draw:
        return "1/2-1/2"
    # A non-draw termination is always a loss for the side to move.
    return "0-1" if state.board.turn == chess.WHITE else "1-0"


def white_score(result: str) -> float:
    return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
