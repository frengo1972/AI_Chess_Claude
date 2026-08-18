"""Stopping and continuing a run must not throw away what it learned.

A network here is meant to grow over many sessions, days apart. That only works
if a resumed run continues four things and not just the weights: the optimizer's
moments, the learning-rate schedule, the random stream, and the counters the KPIs
are built on.
"""

from __future__ import annotations

import json

import pytest
import torch

from app.config import NetworkConfig, RunConfig
from app.engine import train as train_module
from app.engine.train import TRAINER_STATE_FILENAME, TrainingRun
from app.store.metrics import MetricsStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    return MetricsStore(tmp_path / "metrics.db")


def _config() -> RunConfig:
    """The smallest run that still exercises every phase of an iteration."""
    config = RunConfig()
    config.name = "resume"
    config.network = NetworkConfig(history_length=2, residual_blocks=1, filters=8)
    config.search.simulations = 1
    config.selfplay.games_per_iteration = 2
    config.selfplay.workers = 1
    config.selfplay.max_game_plies = 8
    config.selfplay.resign_threshold = None
    config.train.device = "cpu"
    config.train.batch_size = 8
    config.train.steps_per_iteration = 2
    config.train.min_buffer_before_training = 1
    config.train.lr_milestones = (1, 2)
    config.train.lr_gamma = 0.1
    config.arena.enabled = False
    config.benchmark.enabled = False
    return config


def _optimizer_steps(optimizer: torch.optim.Optimizer) -> float:
    return sum(
        float(state["step"]) for state in optimizer.state.values() if "step" in state
    )


def _run(store, run_id: str, *, resume: bool = False, preset: str = "tiny") -> TrainingRun:
    return TrainingRun(
        _config(), run_id=run_id, store=store, preset=preset, resume=resume
    )


# --------------------------------------------------------------------------- #
# The core promise
# --------------------------------------------------------------------------- #


def test_a_second_session_continues_where_the_first_stopped(store):
    first = _run(store, "run-a")
    first.run(iterations=2)

    learning_rate = first.optimizer.param_groups[0]["lr"]
    steps = _optimizer_steps(first.optimizer)
    assert steps > 0, "the first session must have taken gradient steps"

    second = _run(store, "run-a", resume=True)

    assert second.resumed is True
    assert second.start_iteration == 2
    assert second.games_total == first.games_total
    assert second.positions_total == first.positions_total
    assert second.sessions == 2
    assert second.wall_seconds > 0
    assert len(second.buffer) > 0, "the replay shards should refill the buffer"

    # Optimizer moments and the LR schedule survive, so iteration 3 continues
    # rather than restarting with a high learning rate and no momentum.
    assert second.optimizer.param_groups[0]["lr"] == pytest.approx(learning_rate)
    assert _optimizer_steps(second.optimizer) == pytest.approx(steps)
    assert second.scheduler.last_epoch == 2


def test_the_random_stream_is_not_replayed(store):
    """Reseeding would make the next session generate near-duplicate data."""
    first = _run(store, "run-b")
    first.run(iterations=1)

    resumed = _run(store, "run-b", resume=True)
    fresh = _run(store, "run-c")

    assert int(resumed.rng.integers(1 << 30)) != int(fresh.rng.integers(1 << 30))


def test_iterations_continue_numbering_across_sessions(store):
    first = _run(store, "run-d")
    first.run(iterations=2)
    second = _run(store, "run-d", resume=True)
    second.run(iterations=1)

    rows = store.iterations("run-d")
    assert [row["iteration"] for row in rows] == [1, 2, 3]
    assert rows[-1]["wall_seconds"] > rows[0]["wall_seconds"]


def test_elo_and_counters_accumulate_instead_of_resetting(store):
    first = _run(store, "run-e")
    first.run(iterations=1)
    first.elo = 42.0
    first._persist_state(1)

    second = _run(store, "run-e", resume=True)
    assert second.elo == pytest.approx(42.0)
    assert second.games_total == first.games_total


# --------------------------------------------------------------------------- #
# Degraded cases
# --------------------------------------------------------------------------- #


def test_a_run_without_trainer_state_still_resumes(store):
    """Runs from before ``trainer.pt`` existed, or killed mid-first-iteration."""
    first = _run(store, "run-f")
    first.run(iterations=2)
    (first.directory / TRAINER_STATE_FILENAME).unlink()

    second = _run(store, "run-f", resume=True)

    assert second.start_iteration == 2
    # The schedule is fast-forwarded from the iteration count, so the learning
    # rate is not silently reset to its initial value.
    assert second.scheduler.last_epoch == 2
    assert second.optimizer.param_groups[0]["lr"] == pytest.approx(
        _config().train.learning_rate * 0.01
    )
    messages = [event["message"] for event in store.events("run-f")]
    assert any("trainer.pt" in message for message in messages)


def test_a_corrupt_trainer_state_does_not_break_the_resume(store):
    first = _run(store, "run-g")
    first.run(iterations=1)
    (first.directory / TRAINER_STATE_FILENAME).write_bytes(b"not a torch file")

    second = _run(store, "run-g", resume=True)

    assert second.start_iteration == 1
    assert second.optimizer.param_groups[0]["lr"] > 0
    messages = [event["message"] for event in store.events("run-g")]
    assert any("trainer.pt" in message for message in messages)


def test_a_data_collection_iteration_is_persisted_too(store):
    """Otherwise the next session repeats the iterations that only gathered data."""
    config = _config()
    config.train.min_buffer_before_training = 10_000_000
    run = TrainingRun(config, run_id="run-h", store=store)
    run.run(iterations=1)

    state = json.loads((run.directory / "state.json").read_text(encoding="utf-8"))
    assert state["iteration"] == 1
    assert state["games_total"] > 0

    resumed = TrainingRun(config, run_id="run-h", store=store, resume=True)
    assert resumed.start_iteration == 1


def test_resume_flag_without_a_checkpoint_starts_fresh(store):
    run = _run(store, "run-i", resume=True)
    assert run.resumed is False
    assert run.start_iteration == 0
    assert run.best_path.exists()


# --------------------------------------------------------------------------- #
# Run bookkeeping
# --------------------------------------------------------------------------- #


def test_the_run_keeps_its_birth_date_and_preset(store):
    first = _run(store, "run-j", preset="tiny")
    first.run(iterations=1)
    born = store.get_run("run-j")["created_at"]

    _run(store, "run-j", resume=True, preset="small")

    row = store.get_run("run-j")
    assert row["created_at"] == born
    assert row["preset"] == "tiny"


def test_an_older_database_gains_the_new_kpi_columns(tmp_path):
    """History is the point of the project: old databases migrate, not restart."""
    import sqlite3

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE iterations (
                   run_id TEXT NOT NULL, iteration INTEGER NOT NULL,
                   timestamp REAL NOT NULL, games_total INTEGER, elo REAL,
                   extra_json TEXT, PRIMARY KEY (run_id, iteration))"""
        )
        connection.execute(
            "INSERT INTO iterations (run_id, iteration, timestamp, games_total)"
            " VALUES ('old', 1, 0.0, 12)"
        )

    migrated = MetricsStore(path)
    migrated.record_iteration("old", 2, {"games_total": 20, "value_gap": 0.31})

    rows = migrated.iterations("old")
    assert [row["iteration"] for row in rows] == [1, 2]
    assert rows[0]["value_gap"] is None
    assert rows[1]["value_gap"] == pytest.approx(0.31)


def test_watch_settings_survive_a_resume(store):
    from app.engine.watch import WatchSettings, read_settings, write_settings

    first = _run(store, "run-k")
    write_settings(first.watch_dir, WatchSettings(enabled=True, move_delay_ms=300))

    _run(store, "run-k", resume=True)

    settings = read_settings(first.watch_dir)
    assert settings.enabled is True
    assert settings.move_delay_ms == 300
