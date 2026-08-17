"""SQLite-backed KPI store for training runs.

The trainer writes here from its own process; the API reads from the web
process. SQLite in WAL mode handles that pattern without a server, and keeps
every run reproducible after the fact (the exact config is stored with it).
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from app.config import TRAINING_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    preset        TEXT,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    status        TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    network_json  TEXT,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS iterations (
    run_id              TEXT NOT NULL,
    iteration           INTEGER NOT NULL,
    timestamp           REAL NOT NULL,
    games_total         INTEGER,
    positions_total     INTEGER,
    buffer_size         INTEGER,
    policy_loss         REAL,
    value_loss          REAL,
    total_loss          REAL,
    learning_rate       REAL,
    avg_plies           REAL,
    draw_rate           REAL,
    white_win_rate      REAL,
    black_win_rate      REAL,
    resign_rate         REAL,
    policy_entropy      REAL,
    avg_root_value      REAL,
    selfplay_seconds    REAL,
    train_seconds       REAL,
    games_per_minute    REAL,
    positions_per_second REAL,
    arena_score         REAL,
    elo                 REAL,
    elo_delta           REAL,
    promoted            INTEGER,
    benchmark_elo       REAL,
    benchmark_score     REAL,
    extra_json          TEXT,
    PRIMARY KEY (run_id, iteration)
);

CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   TEXT NOT NULL,
    timestamp REAL NOT NULL,
    level    TEXT NOT NULL,
    message  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sample_games (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    result    TEXT,
    plies     INTEGER,
    termination TEXT,
    pgn       TEXT
);

CREATE INDEX IF NOT EXISTS idx_iterations_run ON iterations(run_id, iteration);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_games_run ON sample_games(run_id, iteration DESC);
"""

ITERATION_COLUMNS = [
    "games_total",
    "positions_total",
    "buffer_size",
    "policy_loss",
    "value_loss",
    "total_loss",
    "learning_rate",
    "avg_plies",
    "draw_rate",
    "white_win_rate",
    "black_win_rate",
    "resign_rate",
    "policy_entropy",
    "avg_root_value",
    "selfplay_seconds",
    "train_seconds",
    "games_per_minute",
    "positions_per_second",
    "arena_score",
    "elo",
    "elo_delta",
    "promoted",
    "benchmark_elo",
    "benchmark_score",
]


class MetricsStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or TRAINING_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=15.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            yield connection
            connection.commit()
        finally:
            connection.close()

    # -- runs -------------------------------------------------------------- #

    def create_run(
        self,
        run_id: str,
        name: str,
        config: Dict[str, Any],
        *,
        preset: Optional[str] = None,
        network: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> None:
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO runs
                    (id, name, preset, created_at, updated_at, status, config_json,
                     network_json, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    name,
                    preset,
                    now,
                    now,
                    "running",
                    json.dumps(config),
                    json.dumps(network or {}),
                    notes,
                ),
            )

    def set_status(self, run_id: str, status: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE id = ?",
                (status, time.time(), run_id),
            )

    def list_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_run_row_to_dict(row) for row in rows]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return _run_row_to_dict(row) if row else None

    def latest_run(self) -> Optional[Dict[str, Any]]:
        runs = self.list_runs(limit=1)
        return runs[0] if runs else None

    # -- iterations -------------------------------------------------------- #

    def record_iteration(
        self, run_id: str, iteration: int, metrics: Dict[str, Any]
    ) -> None:
        values = {column: metrics.get(column) for column in ITERATION_COLUMNS}
        extra = {k: v for k, v in metrics.items() if k not in ITERATION_COLUMNS}
        columns = ", ".join(["run_id", "iteration", "timestamp", *ITERATION_COLUMNS, "extra_json"])
        placeholders = ", ".join(["?"] * (3 + len(ITERATION_COLUMNS) + 1))
        with self._connect() as connection:
            connection.execute(
                f"INSERT OR REPLACE INTO iterations ({columns}) VALUES ({placeholders})",
                (
                    run_id,
                    iteration,
                    time.time(),
                    *[values[column] for column in ITERATION_COLUMNS],
                    json.dumps(extra, default=str),
                ),
            )
            connection.execute(
                "UPDATE runs SET updated_at = ? WHERE id = ?", (time.time(), run_id)
            )

    def iterations(self, run_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM iterations WHERE run_id = ? ORDER BY iteration ASC LIMIT ?",
                (run_id, limit),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["extra"] = json.loads(item.pop("extra_json") or "{}")
            result.append(item)
        return result

    def last_iteration(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM iterations WHERE run_id = ? ORDER BY iteration DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["extra"] = json.loads(item.pop("extra_json") or "{}")
        return item

    # -- events & games ---------------------------------------------------- #

    def log(self, run_id: str, message: str, level: str = "info") -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO events (run_id, timestamp, level, message) VALUES (?, ?, ?, ?)",
                (run_id, time.time(), level, message),
            )

    def events(self, run_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY id DESC LIMIT ?",
                (run_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_sample_game(
        self,
        run_id: str,
        iteration: int,
        *,
        result: str,
        plies: int,
        termination: str,
        pgn: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO sample_games (run_id, iteration, result, plies, termination, pgn)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, iteration, result, plies, termination, pgn),
            )
            connection.execute(
                """DELETE FROM sample_games WHERE run_id = ? AND id NOT IN
                   (SELECT id FROM sample_games WHERE run_id = ? ORDER BY id DESC LIMIT 60)""",
                (run_id, run_id),
            )

    def sample_games(self, run_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sample_games WHERE run_id = ? ORDER BY id DESC LIMIT ?",
                (run_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]


def _run_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    item = dict(row)
    item["config"] = json.loads(item.pop("config_json") or "{}")
    item["network"] = json.loads(item.pop("network_json") or "{}")
    return item
