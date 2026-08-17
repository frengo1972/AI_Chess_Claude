"""Stockfish (UCI) wrapper -- the *classical* engine.

Two distinct jobs, deliberately kept in one place so the boundary is obvious:

1. **Human assistance** -- position evaluation and best-move suggestions shown
   in the UI while a *human* plays. See ``analyse``.
2. **Opponent / yardstick** -- a strength-limited opponent the neural network
   can be measured against. See ``best_move``.

What this module must never do is feed the learner. Nothing under
``app.engine`` imports it (``tests/test_engine_isolation.py`` enforces that),
and the training loop only ever consumes it through
``app.services.benchmark``, which reports scores and never writes to the
replay buffer.

``python-chess`` speaks UCI for us; a single engine process is reused and
guarded by a lock, because a UCI process is a single-threaded conversation.
"""

from __future__ import annotations

import os
import platform
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import chess
import chess.engine

from app.config import ENGINES_DIR, SERVER

#: Stockfish's own limits for ``UCI_Elo`` (Stockfish >= 16).
MIN_UCI_ELO = 1320
MAX_UCI_ELO = 3190


class StockfishUnavailable(RuntimeError):
    """Raised when no Stockfish binary can be located."""


@dataclass
class EngineLevel:
    """How strong the classical engine should play.

    ``elo`` uses Stockfish's own strength limiter; ``skill`` uses the coarser
    0-20 skill ladder. ``depth`` / ``movetime`` bound the search.
    """

    elo: Optional[int] = None
    skill: Optional[int] = None
    depth: Optional[int] = 12
    movetime_ms: Optional[int] = None
    threads: int = 1
    hash_mb: int = 64

    def limit(self) -> chess.engine.Limit:
        if self.movetime_ms:
            return chess.engine.Limit(time=self.movetime_ms / 1000.0)
        return chess.engine.Limit(depth=self.depth or 12)

    def options(self) -> Dict[str, object]:
        options: Dict[str, object] = {
            "Threads": max(1, self.threads),
            "Hash": max(16, self.hash_mb),
        }
        if self.elo is not None:
            options["UCI_LimitStrength"] = True
            options["UCI_Elo"] = int(min(max(self.elo, MIN_UCI_ELO), MAX_UCI_ELO))
        elif self.skill is not None:
            options["UCI_LimitStrength"] = False
            options["Skill Level"] = int(min(max(self.skill, 0), 20))
        else:
            options["UCI_LimitStrength"] = False
        return options


@dataclass
class ScoreView:
    """A Stockfish score, expressed the several ways the UI needs."""

    centipawns: Optional[int]
    mate_in: Optional[int]
    white_perspective_cp: Optional[int]
    win_probability: float
    text: str


@dataclass
class AnalysisLine:
    rank: int
    score: ScoreView
    moves_uci: List[str] = field(default_factory=list)
    moves_san: List[str] = field(default_factory=list)
    depth: int = 0
    nodes: int = 0


@dataclass
class AnalysisResult:
    fen: str
    lines: List[AnalysisLine]
    depth: int
    time_ms: int
    engine: str

    @property
    def best_line(self) -> Optional[AnalysisLine]:
        return self.lines[0] if self.lines else None


def find_stockfish() -> Optional[Path]:
    """Locate a Stockfish binary: explicit setting, bundled ``engines/``, PATH."""
    configured = SERVER.stockfish_path or os.environ.get("AICHESS_STOCKFISH_PATH")
    if configured and Path(configured).exists():
        return Path(configured)

    suffix = ".exe" if platform.system() == "Windows" else ""
    if ENGINES_DIR.exists():
        candidates = sorted(ENGINES_DIR.rglob(f"stockfish*{suffix}"))
        for candidate in candidates:
            if candidate.is_file():
                return candidate

    found = shutil.which("stockfish")
    return Path(found) if found else None


class StockfishService:
    """Owns one long-lived Stockfish process, reconfigured per request."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or find_stockfish()
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._lock = threading.Lock()
        self._applied_options: Dict[str, object] = {}
        self._identity: Dict[str, str] = {}

    # -- lifecycle --------------------------------------------------------- #

    @property
    def available(self) -> bool:
        return self._path is not None and Path(self._path).exists()

    @property
    def path(self) -> Optional[str]:
        return str(self._path) if self._path else None

    def info(self) -> Dict[str, object]:
        return {
            "available": self.available,
            "path": self.path,
            "name": self._identity.get("name", "Stockfish" if self.available else None),
            "min_elo": MIN_UCI_ELO,
            "max_elo": MAX_UCI_ELO,
        }

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        if not self.available:
            raise StockfishUnavailable(
                "Stockfish binary not found. Run scripts/download_assets.py, or set "
                "AICHESS_STOCKFISH_PATH."
            )
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(str(self._path))
            self._identity = dict(self._engine.id)
            self._applied_options = {}
        return self._engine

    def _apply(self, engine: chess.engine.SimpleEngine, options: Dict[str, object]) -> None:
        delta = {k: v for k, v in options.items() if self._applied_options.get(k) != v}
        if delta:
            engine.configure(delta)
            self._applied_options.update(delta)

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                try:
                    self._engine.quit()
                except Exception:  # noqa: BLE001 - shutdown must not raise
                    pass
                self._engine = None
                self._applied_options = {}

    def restart(self) -> None:
        self.close()
        self._path = find_stockfish()

    # -- analysis (human assistance) --------------------------------------- #

    def analyse(
        self,
        fen: str,
        *,
        depth: int = 14,
        multipv: int = 3,
        movetime_ms: Optional[int] = None,
        threads: int = 2,
        hash_mb: int = 128,
    ) -> AnalysisResult:
        """Full-strength analysis of ``fen`` with ``multipv`` candidate lines."""
        board = chess.Board(fen)
        level = EngineLevel(
            depth=depth, movetime_ms=movetime_ms, threads=threads, hash_mb=hash_mb
        )
        with self._lock:
            engine = self._ensure_engine()
            self._apply(engine, level.options())
            raw = engine.analyse(
                board,
                level.limit(),
                multipv=max(1, multipv),
                info=chess.engine.INFO_ALL,
            )

        infos = raw if isinstance(raw, list) else [raw]
        lines: List[AnalysisLine] = []
        for rank, info in enumerate(infos, start=1):
            principal = list(info.get("pv", []))
            lines.append(
                AnalysisLine(
                    rank=rank,
                    score=_score_view(info.get("score"), board.turn),
                    moves_uci=[move.uci() for move in principal],
                    moves_san=_san_sequence(board, principal),
                    depth=int(info.get("depth", 0) or 0),
                    nodes=int(info.get("nodes", 0) or 0),
                )
            )

        reached = max((line.depth for line in lines), default=0)
        elapsed = int(float(infos[0].get("time", 0.0) or 0.0) * 1000) if infos else 0
        return AnalysisResult(
            fen=fen,
            lines=lines,
            depth=reached,
            time_ms=elapsed,
            engine=self._identity.get("name", "Stockfish"),
        )

    # -- play (opponent / benchmark) --------------------------------------- #

    def best_move(self, fen: str, level: EngineLevel) -> Optional[chess.Move]:
        """Ask the engine for a move at the requested strength."""
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=False):
            return None
        with self._lock:
            engine = self._ensure_engine()
            self._apply(engine, level.options())
            play = engine.play(board, level.limit())
        return play.move

    def quick_evaluation(self, fen: str, depth: int = 10) -> ScoreView:
        board = chess.Board(fen)
        with self._lock:
            engine = self._ensure_engine()
            self._apply(engine, EngineLevel(depth=depth).options())
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
        return _score_view(info.get("score"), board.turn)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _san_sequence(board: chess.Board, moves: List[chess.Move]) -> List[str]:
    replay = board.copy()
    san: List[str] = []
    for move in moves:
        if move not in replay.legal_moves:
            break
        san.append(replay.san(move))
        replay.push(move)
    return san


def _score_view(score, turn: chess.Color) -> ScoreView:
    if score is None:
        return ScoreView(None, None, None, 0.5, "0.00")

    relative = score.relative
    mate = relative.mate()
    centipawns = relative.score()
    white_cp = score.white().score(mate_score=100_000)

    if mate is not None:
        probability = 1.0 if mate > 0 else 0.0
        text = f"#{mate}" if mate > 0 else f"#-{abs(mate)}"
    else:
        cp = centipawns or 0
        probability = 1.0 / (1.0 + 10 ** (-cp / 400.0))
        pawns = cp / 100.0
        text = f"{pawns:+.2f}"

    return ScoreView(
        centipawns=centipawns,
        mate_in=mate,
        white_perspective_cp=white_cp,
        win_probability=probability,
        text=text,
    )


_SERVICE: Optional[StockfishService] = None


def get_stockfish() -> StockfishService:
    """Process-wide singleton (one UCI process per API server)."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = StockfishService()
    return _SERVICE
