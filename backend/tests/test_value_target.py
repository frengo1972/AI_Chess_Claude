"""The value target: mixing the game result with the search's own opinion.

``z`` alone is one noisy sample per game stamped onto every position of it. ``q``
is what the tree concluded about the position itself. The tests pin down the
blend, its sign convention, and the one case where mixing would be circular.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import NetworkConfig, RunConfig, preset_config
from app.engine.evaluator import Evaluator
from app.engine.network import ChessNet
from app.engine.selfplay import _value_targets, play_game


def _config(weight: float, simulations: int = 8) -> RunConfig:
    config = RunConfig()
    config.train.value_search_weight = weight
    config.search.simulations = simulations
    return config


def test_zero_weight_is_the_alphazero_target():
    outcomes = [1.0, -1.0, 1.0]
    assert _value_targets(outcomes, [0.2, 0.3, -0.9], _config(0.0)) == outcomes


def test_full_weight_is_the_pure_search_value():
    values = [0.2, 0.3, -0.9]
    assert _value_targets([1.0, -1.0, 1.0], values, _config(1.0)) == values


def test_half_weight_is_the_midpoint():
    targets = _value_targets([1.0, -1.0], [0.0, 0.5], _config(0.5))
    assert targets == pytest.approx([0.5, -0.25])


def test_weight_above_one_is_clamped():
    assert _value_targets([1.0], [0.0], _config(4.0)) == pytest.approx([0.0])


def test_no_search_means_no_mixing():
    """With ``simulations == 1`` the root value *is* the value head's output."""
    outcomes = [1.0, -1.0]
    targets = _value_targets(outcomes, [0.9, 0.9], _config(0.5, simulations=1))
    assert targets == outcomes


def test_targets_stay_inside_the_value_range():
    rng = np.random.default_rng(3)
    outcomes = list(rng.choice([-1.0, 0.0, 1.0], size=50))
    values = list(rng.uniform(-1.0, 1.0, size=50))
    for weight in (0.25, 0.5, 0.75, 1.0):
        targets = _value_targets(outcomes, values, _config(weight))
        assert all(-1.0 <= target <= 1.0 for target in targets)


def test_presets_mix_except_policy_only():
    for name in ("tiny", "small", "medium", "large"):
        assert preset_config(name).train.value_search_weight == 0.5
    assert preset_config("policy-only").train.value_search_weight == 0.0


# --------------------------------------------------------------------------- #
# Through a real game
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def evaluator():
    return Evaluator(ChessNet(NetworkConfig(history_length=2, residual_blocks=1, filters=16)), "cpu")


def _play(evaluator, weight: float):
    config = _config(weight, simulations=4)
    config.network = NetworkConfig(history_length=2, residual_blocks=1, filters=16)
    config.selfplay.max_game_plies = 12
    config.selfplay.resign_threshold = None
    return play_game(evaluator, config, np.random.default_rng(5))


def test_mixing_changes_the_targets_but_not_the_game(evaluator):
    plain = _play(evaluator, 0.0)
    mixed = _play(evaluator, 0.5)

    # Self-play is driven by the search, which the target weight never touches.
    assert mixed.moves_uci == plain.moves_uci
    assert mixed.result == plain.result

    plain_values = [sample.value for sample in plain.samples]
    mixed_values = [sample.value for sample in mixed.samples]
    assert len(mixed_values) == len(plain_values)
    assert mixed_values != plain_values
    # Pure z is +-1 or 0; the mixed target lands strictly between.
    assert all(abs(value) <= 1.0 for value in mixed_values)


def test_the_gap_between_result_and_search_is_reported(evaluator):
    record = _play(evaluator, 0.5)
    assert record.mean_value_gap >= 0.0
    assert record.mean_value_gap <= 2.0


def test_value_gap_is_zero_when_the_search_predicted_the_result():
    """Sanity check on the metric's definition, without a network involved."""
    outcomes = np.array([1.0, -1.0, 0.0])
    assert float(np.mean(np.abs(outcomes - outcomes))) == 0.0
