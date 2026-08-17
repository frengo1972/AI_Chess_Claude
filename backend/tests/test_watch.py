"""The spectator feed: it must show the games, and never change them.

Watching self-play is a tap on the training loop, so the tests care about two
things in equal measure: that the boards actually reach the disk, and that a run
produces the very same samples whether anybody is watching or not.
"""

from __future__ import annotations

import json

import chess
import numpy as np
import pytest

from app.config import NetworkConfig, RunConfig
from app.engine.evaluator import Evaluator
from app.engine.network import ChessNet
from app.engine.selfplay import play_game
from app.engine.watch import (
    WINDOW_PLIES,
    WatchPublisher,
    WatchSettings,
    clear_slots,
    material_balance,
    read_settings,
    read_slots,
    watch_directory,
    write_settings,
)


@pytest.fixture
def watch_dir(tmp_path):
    directory = watch_directory(tmp_path / "run-1")
    write_settings(directory, WatchSettings(enabled=True))
    return directory


def _publisher(directory, slot=0):
    return WatchPublisher(directory, slot, run_id="run-1")


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def test_settings_roundtrip(tmp_path):
    directory = watch_directory(tmp_path / "run")
    write_settings(directory, WatchSettings(enabled=True, move_delay_ms=250))
    settings = read_settings(directory)
    assert settings.enabled is True
    assert settings.move_delay_ms == 250
    assert settings.delay_seconds == pytest.approx(0.25)


def test_settings_default_to_off_when_missing_or_broken(tmp_path):
    directory = tmp_path / "nothing-here"
    assert read_settings(directory).enabled is False

    directory.mkdir()
    (directory / "settings.json").write_text("{not json", encoding="utf-8")
    assert read_settings(directory).enabled is False


def test_settings_clamp_an_absurd_delay(tmp_path):
    directory = watch_directory(tmp_path / "run")
    write_settings(directory, WatchSettings.from_dict({"enabled": True, "move_delay_ms": 10_000_000}))
    assert read_settings(directory).move_delay_ms == 5_000


# --------------------------------------------------------------------------- #
# Publisher
# --------------------------------------------------------------------------- #


def test_publisher_writes_nothing_while_disabled(tmp_path):
    directory = watch_directory(tmp_path / "run")
    write_settings(directory, WatchSettings(enabled=False))
    publisher = _publisher(directory)

    board = chess.Board()
    publisher.begin(board, iteration=1, game_index=0)
    board.push_uci("e2e4")
    publisher.record(board, last_move="e2e4", value=0.1)
    publisher.end(board, result="1-0", termination="checkmate")

    assert read_slots(directory) == []


def test_publisher_reports_the_position_and_its_material(watch_dir):
    publisher = _publisher(watch_dir)
    board = chess.Board()
    publisher.begin(board, iteration=3, game_index=2)
    board.push_uci("e2e4")
    publisher.record(board, last_move="e2e4", value=0.42, top_moves=[("e7e5", 0.6)])

    boards = read_slots(watch_dir)
    assert len(boards) == 1
    published = boards[0]
    assert published["slot"] == 0
    assert published["iteration"] == 3
    assert published["game_index"] == 2
    assert published["ply"] == 1
    assert published["turn"] == "black"
    assert published["last_move"] == "e2e4"
    assert published["value"] == pytest.approx(0.42)
    assert published["top_moves"] == [{"uci": "e7e5", "probability": 0.6}]
    assert published["material"] == {"white": 39, "black": 39, "difference": 0}
    assert published["finished"] is False
    assert published["age_seconds"] < 5


def test_publisher_keeps_a_rolling_window_of_positions(watch_dir):
    publisher = _publisher(watch_dir)
    board = chess.Board()
    publisher.begin(board, iteration=1, game_index=0)
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 12:  # 48 legal shuffling plies
        board.push_uci(uci)
        publisher.record(board, last_move=uci, value=0.0)

    published = read_slots(watch_dir)[0]
    frames = published["frames"]
    assert len(frames) == WINDOW_PLIES
    assert [frame["ply"] for frame in frames] == list(range(49 - WINDOW_PLIES, 49))
    assert frames[-1]["fen"] == published["fen"]
    assert published["ply"] == 48


def test_publisher_marks_the_end_of_the_game(watch_dir):
    publisher = _publisher(watch_dir)
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    publisher.begin(board, iteration=1, game_index=0)
    board.push_uci("a1a8")
    publisher.record(board, last_move="a1a8", value=1.0)
    publisher.end(board, result="1-0", termination="checkmate", resigned=False)

    published = read_slots(watch_dir)[0]
    assert published["finished"] is True
    assert published["result"] == "1-0"
    assert published["termination"] == "checkmate"


def test_a_new_game_replaces_the_previous_one_in_the_slot(watch_dir):
    publisher = _publisher(watch_dir)
    board = chess.Board()
    publisher.begin(board, iteration=1, game_index=0)
    board.push_uci("e2e4")
    publisher.record(board, last_move="e2e4", value=0.0)
    publisher.end(board, result="1/2-1/2", termination="draw")

    publisher.begin(chess.Board(), iteration=1, game_index=1)
    published = read_slots(watch_dir)[0]
    assert published["game_index"] == 1
    assert published["finished"] is False
    assert published["ply"] == 0
    assert len(published["frames"]) == 1


# --------------------------------------------------------------------------- #
# Reading side
# --------------------------------------------------------------------------- #


def test_read_slots_is_ordered_and_survives_junk(watch_dir):
    for slot in (2, 0, 1):
        publisher = _publisher(watch_dir, slot=slot)
        publisher.begin(chess.Board(), iteration=1, game_index=slot)
    (watch_dir / "slot-99.json").write_text("half a fi", encoding="utf-8")

    boards = read_slots(watch_dir)
    assert [board["slot"] for board in boards] == [0, 1, 2]


def test_clear_slots_leaves_the_settings_alone(watch_dir):
    _publisher(watch_dir).begin(chess.Board(), iteration=1, game_index=0)
    clear_slots(watch_dir)
    assert read_slots(watch_dir) == []
    assert read_settings(watch_dir).enabled is True


def test_material_balance_counts_pieces_not_kings():
    balance = material_balance(chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1"))
    assert balance == {"white": 9, "black": 0, "difference": 9}


# --------------------------------------------------------------------------- #
# Integration with self-play
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tiny_setup():
    config = RunConfig()
    config.network = NetworkConfig(history_length=2, residual_blocks=1, filters=16)
    config.search.simulations = 1
    config.selfplay.max_game_plies = 10
    config.selfplay.resign_threshold = None
    model = ChessNet(config.network)
    return config, Evaluator(model, "cpu")


def test_self_play_publishes_the_game_it_is_playing(tiny_setup, tmp_path):
    config, evaluator = tiny_setup
    directory = watch_directory(tmp_path / "run")
    write_settings(directory, WatchSettings(enabled=True))

    record = play_game(
        evaluator,
        config,
        np.random.default_rng(7),
        watch=WatchPublisher(directory, 4, run_id="run"),
        iteration=2,
        game_index=5,
    )

    published = read_slots(directory)[0]
    assert published["slot"] == 4
    assert published["iteration"] == 2
    assert published["game_index"] == 5
    assert published["finished"] is True
    assert published["result"] == record.result
    assert published["ply"] == record.plies
    assert published["fen"].split(" ")[0], "a published FEN must describe a real board"


def test_watching_cannot_change_what_the_network_learns(tiny_setup, tmp_path):
    """Same seed, same games -- the feed is an observer, not a participant."""
    config, evaluator = tiny_setup
    directory = watch_directory(tmp_path / "run")
    write_settings(directory, WatchSettings(enabled=True))

    unwatched = play_game(evaluator, config, np.random.default_rng(11))
    watched = play_game(
        evaluator,
        config,
        np.random.default_rng(11),
        watch=WatchPublisher(directory, 0),
    )

    assert watched.moves_uci == unwatched.moves_uci
    assert watched.result == unwatched.result
    assert len(watched.samples) == len(unwatched.samples)
    for left, right in zip(watched.samples, unwatched.samples):
        assert left.value == right.value
        assert np.array_equal(left.bits, right.bits)
        assert np.array_equal(left.policy_probs, right.policy_probs)


def test_slot_files_are_valid_json_at_every_step(watch_dir):
    """The UI polls while the worker writes; it must never read half a file."""
    publisher = _publisher(watch_dir)
    board = chess.Board()
    publisher.begin(board, iteration=1, game_index=0)
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
        board.push_uci(uci)
        publisher.record(board, last_move=uci, value=0.0)
        payload = json.loads((watch_dir / "slot-00.json").read_text(encoding="utf-8"))
        assert payload["ply"] == len(board.move_stack)
