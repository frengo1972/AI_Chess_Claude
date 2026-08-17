"""In-memory game sessions: human vs network, human vs Stockfish, network vs Stockfish.

A session stores the *move list*, not just a FEN, because the neural engine
needs the same position history it was trained on (repetitions, the last T
plies). Everything the UI renders -- legal destinations, SAN, material, PGN --
is derived here so the frontend never has to re-implement the rules.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import chess
import chess.pgn

from app.config import SERVER
from app.engine.rules import terminal_state_from_board
from app.services.nn_engine import EngineMove, get_neural_engine
from app.services.stockfish_service import EngineLevel, get_stockfish

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


class GameError(RuntimeError):
    pass


@dataclass
class OpponentSpec:
    kind: str = "neural"  # "neural" | "stockfish"
    model_id: Optional[str] = None
    simulations: int = 96
    temperature: float = 0.0
    stockfish_elo: Optional[int] = 1500
    stockfish_skill: Optional[int] = None
    stockfish_movetime_ms: int = 200

    def engine_level(self) -> EngineLevel:
        return EngineLevel(
            elo=self.stockfish_elo,
            skill=self.stockfish_skill,
            movetime_ms=self.stockfish_movetime_ms,
            depth=None,
        )


@dataclass
class GameSession:
    id: str
    human_color: Optional[chess.Color]
    opponent: OpponentSpec
    board: chess.Board = field(default_factory=chess.Board)
    moves: List[chess.Move] = field(default_factory=list)
    san_moves: List[str] = field(default_factory=list)
    engine_reports: List[Dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: Optional[str] = None
    termination: Optional[str] = None
    assistance_enabled: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- derived ----------------------------------------------------------- #

    @property
    def moves_uci(self) -> List[str]:
        return [move.uci() for move in self.moves]

    @property
    def is_over(self) -> bool:
        return self.result is not None

    @property
    def human_to_move(self) -> bool:
        return self.human_color is not None and self.board.turn == self.human_color

    def refresh_result(self) -> None:
        termination = terminal_state_from_board(self.board)
        if termination is None:
            return
        if termination.is_draw:
            self.result = "1/2-1/2"
        else:
            self.result = "0-1" if self.board.turn == chess.WHITE else "1-0"
        self.termination = termination.reason

    # -- payloads ---------------------------------------------------------- #

    def legal_destinations(self) -> Dict[str, List[str]]:
        destinations: Dict[str, List[str]] = {}
        for move in self.board.legal_moves:
            origin = chess.square_name(move.from_square)
            destinations.setdefault(origin, []).append(chess.square_name(move.to_square))
        return {k: sorted(set(v)) for k, v in destinations.items()}

    def promotion_squares(self) -> List[str]:
        return sorted(
            {
                chess.square_name(move.from_square) + chess.square_name(move.to_square)
                for move in self.board.legal_moves
                if move.promotion is not None
            }
        )

    def material(self) -> Dict[str, int]:
        white = sum(
            PIECE_VALUES.get(piece.piece_type, 0)
            for piece in self.board.piece_map().values()
            if piece.color == chess.WHITE
        )
        black = sum(
            PIECE_VALUES.get(piece.piece_type, 0)
            for piece in self.board.piece_map().values()
            if piece.color == chess.BLACK
        )
        return {"white": white, "black": black, "difference": white - black}

    def captured(self) -> Dict[str, List[str]]:
        start_counts = {
            chess.WHITE: {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1},
            chess.BLACK: {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1},
        }
        for piece in self.board.piece_map().values():
            if piece.piece_type in start_counts[piece.color]:
                start_counts[piece.color][piece.piece_type] -= 1
        return {
            "white": _expand_captures(start_counts[chess.WHITE]),
            "black": _expand_captures(start_counts[chess.BLACK]),
        }

    def pgn(self) -> str:
        game = chess.pgn.Game()
        game.headers["Event"] = "AI_Chess_Claude"
        game.headers["White"] = self._player_name(chess.WHITE)
        game.headers["Black"] = self._player_name(chess.BLACK)
        game.headers["Result"] = self.result or "*"
        node = game
        for move in self.moves:
            node = node.add_variation(move)
        return str(game)

    def _player_name(self, colour: chess.Color) -> str:
        if self.human_color == colour:
            return "Human"
        if self.opponent.kind == "stockfish":
            elo = self.opponent.stockfish_elo
            return f"Stockfish ({elo} Elo)" if elo else "Stockfish"
        return f"NeuralNet ({self.opponent.model_id})"

    def to_payload(self) -> Dict:
        return {
            "id": self.id,
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "human_color": _colour_name(self.human_color),
            "human_to_move": self.human_to_move,
            "opponent": {
                "kind": self.opponent.kind,
                "model_id": self.opponent.model_id,
                "simulations": self.opponent.simulations,
                "stockfish_elo": self.opponent.stockfish_elo,
                "stockfish_skill": self.opponent.stockfish_skill,
            },
            "assistance_enabled": self.assistance_enabled,
            "legal_moves": self.legal_destinations(),
            "promotion_moves": self.promotion_squares(),
            "moves_uci": self.moves_uci,
            "moves_san": self.san_moves,
            "last_move": self.moves[-1].uci() if self.moves else None,
            "in_check": self.board.is_check(),
            "checkers": [chess.square_name(s) for s in self.board.checkers()],
            "is_over": self.is_over,
            "result": self.result,
            "termination": self.termination,
            "material": self.material(),
            "captured": self.captured(),
            "halfmove_clock": self.board.halfmove_clock,
            "fullmove_number": self.board.fullmove_number,
            "engine_reports": self.engine_reports[-12:],
            "pgn": self.pgn(),
            "updated_at": self.updated_at,
        }


def _expand_captures(counts: Dict[int, int]) -> List[str]:
    symbols = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q"}
    out: List[str] = []
    for piece_type, missing in counts.items():
        out.extend([symbols[piece_type]] * max(0, missing))
    return out


def _colour_name(colour: Optional[chess.Color]) -> Optional[str]:
    if colour is None:
        return None
    return "white" if colour == chess.WHITE else "black"


class GameManager:
    """Process-wide registry of live games."""

    def __init__(self) -> None:
        self._games: Dict[str, GameSession] = {}
        self._lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------- #

    def create(
        self,
        *,
        human_color: Optional[str] = "white",
        opponent: Optional[OpponentSpec] = None,
        assistance_enabled: bool = True,
        starting_fen: Optional[str] = None,
    ) -> GameSession:
        self._evict_stale()
        spec = opponent or OpponentSpec()
        if spec.kind == "neural" and not spec.model_id:
            spec.model_id = get_neural_engine().default_model_id()

        colour: Optional[chess.Color]
        if human_color in (None, "none", "spectator"):
            colour = None
        elif human_color == "black":
            colour = chess.BLACK
        else:
            colour = chess.WHITE

        session = GameSession(
            id=uuid.uuid4().hex[:12],
            human_color=colour,
            opponent=spec,
            board=chess.Board(starting_fen) if starting_fen else chess.Board(),
            assistance_enabled=assistance_enabled,
        )
        with self._lock:
            self._games[session.id] = session
        return session

    def get(self, game_id: str) -> GameSession:
        with self._lock:
            session = self._games.get(game_id)
        if session is None:
            raise GameError(f"unknown game {game_id!r}")
        return session

    def delete(self, game_id: str) -> None:
        with self._lock:
            self._games.pop(game_id, None)

    def _evict_stale(self) -> None:
        cutoff = time.time() - SERVER.game_ttl_seconds
        with self._lock:
            stale = [k for k, v in self._games.items() if v.updated_at < cutoff]
            for key in stale:
                self._games.pop(key, None)
            if len(self._games) > SERVER.max_concurrent_games:
                oldest = sorted(self._games.items(), key=lambda kv: kv[1].updated_at)
                for key, _ in oldest[: len(self._games) - SERVER.max_concurrent_games]:
                    self._games.pop(key, None)

    # -- moves ------------------------------------------------------------- #

    def apply_move(self, session: GameSession, uci: str) -> GameSession:
        with session.lock:
            if session.is_over:
                raise GameError("the game is already over")
            try:
                move = chess.Move.from_uci(uci)
            except ValueError as error:
                raise GameError(f"malformed move {uci!r}") from error
            if move not in session.board.legal_moves:
                raise GameError(f"illegal move {uci!r} in this position")
            session.san_moves.append(session.board.san(move))
            session.board.push(move)
            session.moves.append(move)
            session.updated_at = time.time()
            session.refresh_result()
        return session

    def undo(self, session: GameSession, plies: int = 2) -> GameSession:
        with session.lock:
            for _ in range(min(plies, len(session.moves))):
                session.board.pop()
                session.moves.pop()
                session.san_moves.pop()
            session.result = None
            session.termination = None
            session.engine_reports = session.engine_reports[: len(session.moves)]
            session.updated_at = time.time()
        return session

    def resign(self, session: GameSession) -> GameSession:
        with session.lock:
            if not session.is_over:
                loser = session.human_color if session.human_color is not None else session.board.turn
                session.result = "0-1" if loser == chess.WHITE else "1-0"
                session.termination = "resignation"
                session.updated_at = time.time()
        return session

    # -- engine turn ------------------------------------------------------- #

    def engine_move(self, session: GameSession) -> Optional[Dict]:
        """Let the non-human side move. Returns the engine's report."""
        with session.lock:
            if session.is_over:
                return None
            if session.human_color is not None and session.board.turn == session.human_color:
                raise GameError("it is the human's turn")

            spec = session.opponent
            if spec.kind == "stockfish":
                report = self._stockfish_turn(session, spec)
            else:
                report = self._neural_turn(session, spec)

            if report is None:
                session.refresh_result()
                return None

            move = chess.Move.from_uci(report["uci"])
            session.san_moves.append(session.board.san(move))
            session.board.push(move)
            session.moves.append(move)
            session.engine_reports.append(report)
            session.updated_at = time.time()
            session.refresh_result()
            return report

    def _neural_turn(self, session: GameSession, spec: OpponentSpec) -> Optional[Dict]:
        engine = get_neural_engine()
        result: Optional[EngineMove] = engine.choose_move(
            session.moves_uci,
            model_id=spec.model_id or engine.default_model_id(),
            simulations=spec.simulations,
            temperature=spec.temperature,
        )
        if result is None:
            return None
        return {
            "source": "neural",
            "uci": result.move_uci,
            "san": result.move_san,
            "evaluation": round(result.evaluation, 4),
            "simulations": result.simulations,
            "nodes": result.nodes,
            "think_ms": result.think_ms,
            "top_moves": result.top_moves,
            "principal_variation": result.principal_variation,
            "model_id": result.model_id,
        }

    def _stockfish_turn(self, session: GameSession, spec: OpponentSpec) -> Optional[Dict]:
        stockfish = get_stockfish()
        move = stockfish.best_move(session.board.fen(), spec.engine_level())
        if move is None:
            return None
        return {
            "source": "stockfish",
            "uci": move.uci(),
            "san": session.board.san(move),
            "elo": spec.stockfish_elo,
            "skill": spec.stockfish_skill,
            "think_ms": spec.stockfish_movetime_ms,
        }


_MANAGER: Optional[GameManager] = None


def get_game_manager() -> GameManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = GameManager()
    return _MANAGER
