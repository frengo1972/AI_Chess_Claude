"""Live window onto self-play, for the human watching the training.

Self-play runs in worker processes spawned by the trainer, which is itself a
separate OS process from the web server. There is no shared memory to look
into, so the workers publish what they are doing the same way the trainer
publishes KPIs: through the filesystem. Each worker owns one *slot* file and
rewrites it after every move; the API reads whatever is on disk and pushes it to
the browser.

Three deliberate choices:

* **Opt-in, and switchable at runtime.** Publishing costs a small write per
  ply, and pacing the moves costs real training throughput. Both are driven by a
  settings file that the workers re-read a few times a second, so the switch can
  be flipped on a run that is already going, without restarting it.
* **A rolling window of positions, not just the latest one.** The browser paces
  the moves itself, so the games stay watchable while the trainer keeps running
  at full speed. To do that it needs the positions it has not shown yet; the
  last ~20 are enough, and a client that falls further behind simply jumps
  forward.
* **No new IPC.** A slot file is rewritten in full and moved into place with
  ``os.replace``, which is atomic on Windows too, so a reader either sees the
  previous snapshot or the next one, never half of either.

This module stays inside the isolation boundary enforced by
``tests/test_engine_isolation.py``: standard library plus ``chess``, nothing
else. It is *output* from the engine, never input to it — nothing published here
is ever read back into training.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import chess

WATCH_DIRNAME = "watch"
SETTINGS_FILENAME = "settings.json"
SLOT_PATTERN = "slot-*.json"

WINDOW_PLIES = 20
"""Positions kept in a slot file, so the browser has something to animate."""

SETTINGS_POLL_SECONDS = 0.5
"""How often a worker re-reads the settings file. Cheap enough to be constant."""

MAX_DELAY_MS = 5_000

_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


@dataclass
class WatchSettings:
    """What the observer asked for. Lives in ``<run>/watch/settings.json``."""

    enabled: bool = False
    move_delay_ms: int = 0
    """Pause after each self-play move. Non-zero slows training down by design."""

    @property
    def delay_seconds(self) -> float:
        return max(0, min(int(self.move_delay_ms), MAX_DELAY_MS)) / 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {"enabled": bool(self.enabled), "move_delay_ms": int(self.move_delay_ms)}

    @classmethod
    def from_dict(cls, payload: Any) -> "WatchSettings":
        if not isinstance(payload, dict):
            return cls()
        try:
            delay = int(payload.get("move_delay_ms") or 0)
        except (TypeError, ValueError):
            delay = 0
        return cls(
            enabled=bool(payload.get("enabled", False)),
            move_delay_ms=max(0, min(delay, MAX_DELAY_MS)),
        )


def watch_directory(run_directory: Path) -> Path:
    return Path(run_directory) / WATCH_DIRNAME


def read_settings(watch_dir: Path) -> WatchSettings:
    path = Path(watch_dir) / SETTINGS_FILENAME
    try:
        return WatchSettings.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return WatchSettings()


def write_settings(watch_dir: Path, settings: WatchSettings) -> WatchSettings:
    directory = Path(watch_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_write(directory / SETTINGS_FILENAME, settings.to_dict())
    return settings


def clear_slots(watch_dir: Path) -> None:
    """Drop stale boards, so a new run never shows the previous one's games."""
    for path in Path(watch_dir).glob(SLOT_PATTERN):
        try:
            path.unlink()
        except OSError:
            pass


def read_slots(watch_dir: Path) -> List[Dict[str, Any]]:
    """Every readable slot snapshot, oldest slot first, annotated with its age."""
    now = time.time()
    boards: List[Dict[str, Any]] = []
    for path in sorted(Path(watch_dir).glob(SLOT_PATTERN)):
        try:
            board = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # mid-write or truncated: it will be there next poll
        if not isinstance(board, dict):
            continue
        board["age_seconds"] = round(max(0.0, now - float(board.get("updated_at") or now)), 2)
        boards.append(board)
    boards.sort(key=lambda item: item.get("slot", 0))
    return boards


def material_balance(board: chess.Board) -> Dict[str, int]:
    white = black = 0
    for piece_type, value in _PIECE_VALUES.items():
        white += value * len(board.pieces(piece_type, chess.WHITE))
        black += value * len(board.pieces(piece_type, chess.BLACK))
    return {"white": white, "black": black, "difference": white - black}


# --------------------------------------------------------------------------- #
# Publisher (worker side)
# --------------------------------------------------------------------------- #


class WatchPublisher:
    """One per self-play worker; owns exactly one slot file.

    Every method is a no-op while watching is disabled, so the only cost of
    leaving the feature wired in is one small file read every
    ``SETTINGS_POLL_SECONDS``.
    """

    def __init__(
        self,
        watch_dir: Path,
        slot: int,
        *,
        run_id: Optional[str] = None,
        window: int = WINDOW_PLIES,
    ) -> None:
        self.directory = Path(watch_dir)
        self.slot = int(slot)
        self.run_id = run_id
        self.window = max(1, int(window))
        self.path = self.directory / f"slot-{self.slot:02d}.json"

        self._settings = WatchSettings()
        self._checked_at = 0.0
        self._frames: List[Dict[str, Any]] = []
        self._state: Dict[str, Any] = {}

    # -- settings ---------------------------------------------------------- #

    @property
    def settings(self) -> WatchSettings:
        now = time.monotonic()
        if now - self._checked_at >= SETTINGS_POLL_SECONDS:
            self._checked_at = now
            self._settings = read_settings(self.directory)
        return self._settings

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def pace(self) -> None:
        """Hold the position on screen long enough for a human to read it."""
        delay = self._settings.delay_seconds
        if delay > 0 and self._settings.enabled:
            time.sleep(delay)

    # -- game lifecycle ----------------------------------------------------- #

    def begin(self, board: chess.Board, *, iteration: int, game_index: int) -> None:
        if not self.enabled:
            return
        self._frames = []
        self._state = {
            "slot": self.slot,
            "run_id": self.run_id,
            "iteration": int(iteration),
            "game_index": int(game_index),
            "game_uid": f"{self.slot}-{iteration}-{game_index}",
            "started_at": time.time(),
            "finished": False,
            "result": None,
            "termination": None,
            "resigned": False,
        }
        self.record(board, last_move=None, value=None, top_moves=())

    def record(
        self,
        board: chess.Board,
        *,
        last_move: Optional[str],
        value: Optional[float],
        top_moves: Any = (),
    ) -> None:
        """Publish the position now on the board.

        ``value`` is the search value of the move that produced it, from the
        point of view of the side that played it -- that is what a spectator
        wants to read next to the board.
        """
        if not self.enabled or not self._state:
            return
        frame = {
            "ply": len(board.move_stack),
            "fen": board.fen(),
            "last_move": last_move,
            "value": None if value is None else round(float(value), 4),
        }
        self._frames.append(frame)
        del self._frames[: max(0, len(self._frames) - self.window)]

        self._state.update(
            {
                "ply": frame["ply"],
                "fen": frame["fen"],
                "turn": "white" if board.turn == chess.WHITE else "black",
                "last_move": last_move,
                "value": frame["value"],
                "in_check": board.is_check(),
                "material": material_balance(board),
                "top_moves": [
                    {"uci": uci, "probability": round(float(probability), 4)}
                    for uci, probability in top_moves
                ],
            }
        )
        self._flush()

    def end(
        self,
        board: chess.Board,
        *,
        result: str,
        termination: str,
        resigned: bool = False,
    ) -> None:
        if not self.enabled or not self._state:
            return
        self._state.update(
            {
                "finished": True,
                "result": result,
                "termination": termination,
                "resigned": bool(resigned),
                "plies": len(board.move_stack),
            }
        )
        self._flush()

    # -- io ----------------------------------------------------------------- #

    def _flush(self) -> None:
        payload = dict(self._state)
        payload["frames"] = list(self._frames)
        payload["updated_at"] = time.time()
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            _atomic_write(self.path, payload)
        except OSError:
            pass  # a spectator feed must never be able to break a training run


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, default=str), encoding="utf-8")
    os.replace(temporary, path)
