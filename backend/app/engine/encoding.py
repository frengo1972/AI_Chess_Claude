"""Board <-> tensor encoding, and move <-> policy-index encoding.

This module is the contract between the rules of chess (handled by
``python-chess``) and the neural network. It follows the AlphaZero conventions
described in *Mastering Chess and Shogi by Self-Play with a General
Reinforcement Learning Algorithm* (Silver et al., 2017), with the history
length left configurable so small experiments stay cheap.

Input planes
------------
Everything is expressed from the point of view of the side to move ("the
mover"). When Black is to move the board is mirrored vertically and the piece
colours are swapped, so the network only ever has to learn "how to play as the
side moving up the board".

For each of ``T`` history steps (step 0 = current position), 14 planes::

    +0 .. +5    mover's   P N B R Q K
    +6 .. +11   opponent's P N B R Q K
    +12         1.0 if this position had already occurred once before
    +13         1.0 if this position had already occurred twice before

Then 7 constant planes::

    +0          1.0 if the mover is White (absolute colour)
    +1          fullmove number / 100
    +2          mover can castle kingside
    +3          mover can castle queenside
    +4          opponent can castle kingside
    +5          opponent can castle queenside
    +6          halfmove clock (50-move rule counter) / 100

Total: ``14 * T + 7`` planes of 8x8. With ``T = 8`` this is the canonical 119.

Policy output
-------------
``73 * 64 = 4672`` logits, laid out as ``plane * 64 + from_square`` so the
tensor can be produced directly by a convolution with 73 output channels and
flattened in ``(C, rank, file)`` order.

The 73 planes are::

    0  .. 55    "queen" moves: 8 directions x 7 distances
    56 .. 63    knight moves
    64 .. 72    underpromotions: 3 pieces (N, B, R) x 3 directions (left, straight, right)

Queen promotions are *not* given their own planes: they are encoded as the
ordinary one-square forward (or diagonal) pawn move, exactly as in AlphaZero.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import chess
import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PLANES_PER_STEP = 14
CONSTANT_PLANES = 7
NUM_SQUARES = 64
NUM_MOVE_PLANES = 73
POLICY_SIZE = NUM_MOVE_PLANES * NUM_SQUARES  # 4672

#: Piece types in the order used by the piece planes.
PIECE_ORDER: Tuple[int, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)

#: (delta_rank, delta_file) for the 8 "queen" directions, in a fixed order.
QUEEN_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (1, 0),    # N
    (1, 1),    # NE
    (0, 1),    # E
    (-1, 1),   # SE
    (-1, 0),   # S
    (-1, -1),  # SW
    (0, -1),   # W
    (1, -1),   # NW
)

#: (delta_rank, delta_file) for the 8 knight moves, in a fixed order.
KNIGHT_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
)

#: Underpromotion targets, in plane order.
UNDERPROMOTION_PIECES: Tuple[int, ...] = (chess.KNIGHT, chess.BISHOP, chess.ROOK)

_QUEEN_DIRECTION_INDEX = {d: i for i, d in enumerate(QUEEN_DIRECTIONS)}
_KNIGHT_DIRECTION_INDEX = {d: i for i, d in enumerate(KNIGHT_DIRECTIONS)}
_UNDERPROMOTION_INDEX = {p: i for i, p in enumerate(UNDERPROMOTION_PIECES)}

_QUEEN_PLANE_END = 56
_KNIGHT_PLANE_END = 64


def num_input_planes(history_length: int) -> int:
    """Number of 8x8 input planes for a given history length."""
    return PLANES_PER_STEP * history_length + CONSTANT_PLANES


# --------------------------------------------------------------------------- #
# Move <-> policy index
# --------------------------------------------------------------------------- #


def _orient_square(square: int, white_to_move: bool) -> int:
    """Map an absolute square into the mover-relative frame (and back).

    The transformation is its own inverse: mirroring vertically is ``sq ^ 56``.
    """
    return square if white_to_move else square ^ 56


def move_to_index(move: chess.Move, white_to_move: bool) -> int:
    """Encode a legal move into its ``[0, 4672)`` policy index.

    ``white_to_move`` selects the orientation; it must match the side that is
    actually making ``move``.
    """
    from_sq = _orient_square(move.from_square, white_to_move)
    to_sq = _orient_square(move.to_square, white_to_move)

    from_rank, from_file = divmod(from_sq, 8)
    to_rank, to_file = divmod(to_sq, 8)
    d_rank = to_rank - from_rank
    d_file = to_file - from_file

    promotion = move.promotion
    if promotion is not None and promotion != chess.QUEEN:
        if promotion not in _UNDERPROMOTION_INDEX:
            raise ValueError(f"unsupported promotion piece: {promotion}")
        if d_rank != 1 or abs(d_file) > 1:
            raise ValueError(f"malformed underpromotion move: {move}")
        plane = (
            _KNIGHT_PLANE_END
            + _UNDERPROMOTION_INDEX[promotion] * 3
            + (d_file + 1)
        )
        return plane * NUM_SQUARES + from_sq

    knight_index = _KNIGHT_DIRECTION_INDEX.get((d_rank, d_file))
    if knight_index is not None:
        plane = _QUEEN_PLANE_END + knight_index
        return plane * NUM_SQUARES + from_sq

    distance = max(abs(d_rank), abs(d_file))
    if distance == 0 or distance > 7:
        raise ValueError(f"move with impossible geometry: {move}")
    direction = (
        (d_rank > 0) - (d_rank < 0),
        (d_file > 0) - (d_file < 0),
    )
    direction_index = _QUEEN_DIRECTION_INDEX.get(direction)
    if direction_index is None:
        raise ValueError(f"move is neither a queen nor a knight move: {move}")
    if (
        direction[0] * distance != d_rank
        or direction[1] * distance != d_file
    ):
        raise ValueError(f"move is not a straight ray: {move}")

    plane = direction_index * 7 + (distance - 1)
    return plane * NUM_SQUARES + from_sq


def index_to_move(index: int, board: chess.Board) -> Optional[chess.Move]:
    """Decode a policy index back into a move for ``board``.

    Returns ``None`` when the index does not correspond to a geometrically
    valid move on this board. The result is *not* checked for legality; callers
    that need legality should test it against ``board.legal_moves`` (or use
    :func:`legal_move_indices`, which is both faster and safer).
    """
    if not 0 <= index < POLICY_SIZE:
        return None

    white_to_move = board.turn == chess.WHITE
    plane, from_sq_oriented = divmod(index, NUM_SQUARES)
    from_rank, from_file = divmod(from_sq_oriented, 8)

    promotion: Optional[int] = None

    if plane < _QUEEN_PLANE_END:
        direction_index, distance_index = divmod(plane, 7)
        d_rank, d_file = QUEEN_DIRECTIONS[direction_index]
        distance = distance_index + 1
        to_rank = from_rank + d_rank * distance
        to_file = from_file + d_file * distance
    elif plane < _KNIGHT_PLANE_END:
        d_rank, d_file = KNIGHT_DIRECTIONS[plane - _QUEEN_PLANE_END]
        to_rank = from_rank + d_rank
        to_file = from_file + d_file
    else:
        offset = plane - _KNIGHT_PLANE_END
        piece_index, direction_index = divmod(offset, 3)
        promotion = UNDERPROMOTION_PIECES[piece_index]
        to_rank = from_rank + 1
        to_file = from_file + (direction_index - 1)

    if not (0 <= to_rank < 8 and 0 <= to_file < 8):
        return None

    from_square = _orient_square(from_sq_oriented, white_to_move)
    to_square = _orient_square(to_rank * 8 + to_file, white_to_move)

    if promotion is None:
        piece = board.piece_at(from_square)
        if piece is not None and piece.piece_type == chess.PAWN:
            # In the oriented frame a promotion always lands on rank index 7.
            if to_rank == 7:
                promotion = chess.QUEEN

    return chess.Move(from_square, to_square, promotion=promotion)


def legal_move_indices(board: chess.Board) -> Dict[int, chess.Move]:
    """Map every legal move of ``board`` to its policy index.

    This is *the* move-generation step of the engine: the network never gets to
    consider anything that is not produced here, so the rules of chess are
    enforced structurally rather than learned.
    """
    white_to_move = board.turn == chess.WHITE
    mapping: Dict[int, chess.Move] = {}
    for move in board.legal_moves:
        mapping[move_to_index(move, white_to_move)] = move
    return mapping


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Boolean mask of shape ``(4672,)`` with ``True`` on legal moves."""
    mask = np.zeros(POLICY_SIZE, dtype=bool)
    white_to_move = board.turn == chess.WHITE
    for move in board.legal_moves:
        mask[move_to_index(move, white_to_move)] = True
    return mask


# --------------------------------------------------------------------------- #
# Position -> input planes
# --------------------------------------------------------------------------- #


def _absolute_step_planes(board: chess.Board, repetitions: int) -> np.ndarray:
    """The 14 per-step planes for ``board``, in the absolute (White-up) frame.

    Planes 0-5 are White's pieces, 6-11 Black's, 12-13 the repetition flags.
    Orientation is applied later, once, by :func:`orient_step_planes`.
    """
    planes = np.zeros((PLANES_PER_STEP, 8, 8), dtype=np.float32)
    for piece_index, piece_type in enumerate(PIECE_ORDER):
        for square in board.pieces(piece_type, chess.WHITE):
            planes[piece_index, square >> 3, square & 7] = 1.0
        for square in board.pieces(piece_type, chess.BLACK):
            planes[piece_index + 6, square >> 3, square & 7] = 1.0
    if repetitions >= 1:
        planes[12].fill(1.0)
    if repetitions >= 2:
        planes[13].fill(1.0)
    return planes


def orient_step_planes(planes: np.ndarray, white_to_move: bool) -> np.ndarray:
    """Rotate absolute-frame step planes into the mover-relative frame."""
    if white_to_move:
        return planes
    flipped = planes[:, ::-1, :]
    oriented = np.empty_like(flipped)
    oriented[0:6] = flipped[6:12]
    oriented[6:12] = flipped[0:6]
    oriented[12:14] = flipped[12:14]
    return oriented


def _constant_planes(board: chess.Board) -> np.ndarray:
    """The 7 scalar-broadcast planes describing side, clocks and castling."""
    planes = np.zeros((CONSTANT_PLANES, 8, 8), dtype=np.float32)
    mover = board.turn
    opponent = not mover

    if mover == chess.WHITE:
        planes[0].fill(1.0)
    planes[1].fill(min(board.fullmove_number, 500) / 100.0)
    if board.has_kingside_castling_rights(mover):
        planes[2].fill(1.0)
    if board.has_queenside_castling_rights(mover):
        planes[3].fill(1.0)
    if board.has_kingside_castling_rights(opponent):
        planes[4].fill(1.0)
    if board.has_queenside_castling_rights(opponent):
        planes[5].fill(1.0)
    planes[6].fill(min(board.halfmove_clock, 100) / 100.0)
    return planes


class PositionHistory:
    """A board plus the rolling history the network is fed.

    The class owns a live :class:`chess.Board` and keeps, for every ply played
    so far, the pre-computed 14 absolute-frame planes. Absolute-frame planes are
    orientation-independent, so they can be cached once and simply flipped at
    encode time. ``push`` appends and ``pop`` truncates, both O(1) -- which
    matters because MCTS does it hundreds of times per move.
    """

    __slots__ = ("board", "history_length", "_planes", "_repetitions", "_undo")

    def __init__(
        self,
        board: Optional[chess.Board] = None,
        history_length: int = 8,
    ) -> None:
        if history_length < 1:
            raise ValueError("history_length must be >= 1")
        self.board = board.copy() if board is not None else chess.Board()
        self.history_length = history_length
        self._repetitions: Counter = Counter()
        self._planes: List[np.ndarray] = []
        self._undo: List[int] = []

        key = self.board._transposition_key()  # noqa: SLF001 - fast repetition key
        self._repetitions[key] = 1
        self._planes.append(_absolute_step_planes(self.board, 0))

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_moves(
        cls,
        moves: Iterable[chess.Move],
        *,
        root: Optional[chess.Board] = None,
        history_length: int = 8,
    ) -> "PositionHistory":
        """Replay ``moves`` from ``root`` so the history is properly populated."""
        state = cls(root, history_length=history_length)
        for move in moves:
            state.push(move)
        return state

    def copy(self) -> "PositionHistory":
        clone = PositionHistory.__new__(PositionHistory)
        clone.board = self.board.copy()
        clone.history_length = self.history_length
        clone._repetitions = self._repetitions.copy()
        clone._planes = list(self._planes)  # planes themselves are never mutated
        clone._undo = list(self._undo)
        return clone

    # -- mutation ---------------------------------------------------------- #

    def push(self, move: chess.Move) -> None:
        self.board.push(move)
        key = self.board._transposition_key()  # noqa: SLF001
        earlier_occurrences = self._repetitions[key]
        self._repetitions[key] = earlier_occurrences + 1
        self._undo.append(earlier_occurrences)
        self._planes.append(_absolute_step_planes(self.board, earlier_occurrences))

    def pop(self) -> chess.Move:
        earlier_occurrences = self._undo.pop()
        key = self.board._transposition_key()  # noqa: SLF001
        if earlier_occurrences:
            self._repetitions[key] = earlier_occurrences
        else:
            del self._repetitions[key]
        self._planes.pop()
        return self.board.pop()

    # -- queries ----------------------------------------------------------- #

    def repetition_count(self) -> int:
        """How many times the current position has occurred (1 = first time)."""
        return self._repetitions[self.board._transposition_key()]  # noqa: SLF001

    @property
    def ply(self) -> int:
        return len(self._undo)

    # -- encoding ---------------------------------------------------------- #

    def encode(self) -> np.ndarray:
        """Return the ``(14*T + 7, 8, 8)`` float32 input tensor."""
        white_to_move = self.board.turn == chess.WHITE
        total = num_input_planes(self.history_length)
        out = np.zeros((total, 8, 8), dtype=np.float32)

        window = self._planes[-self.history_length :]
        # ``window`` is oldest-first; step 0 of the encoding is the newest.
        for age, absolute in enumerate(reversed(window)):
            offset = age * PLANES_PER_STEP
            out[offset : offset + PLANES_PER_STEP] = orient_step_planes(
                absolute, white_to_move
            )

        out[PLANES_PER_STEP * self.history_length :] = _constant_planes(self.board)
        return out


def encode_board(board: chess.Board, history_length: int = 8) -> np.ndarray:
    """Encode a bare board (its own move stack supplies the history)."""
    root = board.root()
    state = PositionHistory(root, history_length=history_length)
    for move in board.move_stack:
        state.push(move)
    return state.encode()
