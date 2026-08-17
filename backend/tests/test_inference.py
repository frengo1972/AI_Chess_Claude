"""The shared inference server: same answers as a local model, or a clean error.

Moving the network out of the worker is only acceptable if nothing about the
game changes. These tests pin that down (identical outputs, identical self-play),
plus the two failure modes that matter in a long run: a client must never hang
forever, and a missing GPU must fall back instead of crashing.
"""

from __future__ import annotations

import multiprocessing as mp
import queue

import numpy as np
import pytest
import torch

from app.config import NetworkConfig, RunConfig, preset_config
from app.engine.evaluator import Evaluator
from app.engine.inference import InferenceServer, RemoteEvaluator
from app.engine.network import ChessNet, save_checkpoint
from app.engine.selfplay import _start_inference_server, generate_selfplay, play_game

NETWORK = NetworkConfig(history_length=2, residual_blocks=1, filters=16)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("inference") / "best.pt"
    torch.manual_seed(0)
    save_checkpoint(path, ChessNet(NETWORK), iteration=0)
    return path


@pytest.fixture
def server(checkpoint):
    """A server on the CPU: the transport is what is under test, not the device."""
    context = mp.get_context("spawn")
    running = InferenceServer(
        checkpoint,
        device="cpu",
        max_batch=64,
        collect_timeout_ms=1.0,
        request_queue=context.Queue(),
        response_queues=[context.Queue() for _ in range(2)],
        counter=context.Value("i", 0),
    ).start()
    yield running
    running.stop()


def _client(server, index=0, **kwargs):
    return RemoteEvaluator(index, server.requests, server.responses[index], **kwargs)


def _planes(count, seed=0):
    rng = np.random.default_rng(seed)
    planes = NETWORK.history_length * 14 + 7
    return [rng.random((planes, 8, 8), dtype=np.float32) for _ in range(count)]


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #


def test_remote_answers_match_a_local_model(server, checkpoint):
    local = Evaluator(*_load(checkpoint))
    remote = _client(server)
    positions = _planes(5, seed=1)

    for planes, (logits, value) in zip(positions, remote.evaluate_many(positions)):
        expected_logits, expected_value = local.evaluate(planes)
        assert np.allclose(logits, expected_logits, atol=1e-5)
        assert value == pytest.approx(expected_value, abs=1e-5)


def _load(checkpoint):
    from app.engine.network import load_checkpoint

    model, _ = load_checkpoint(checkpoint, device="cpu")
    return model, "cpu"


def test_two_clients_never_read_each_others_answers(server):
    first, second = _client(server, 0), _client(server, 1)
    left = _planes(3, seed=2)
    right = _planes(3, seed=3)

    left_out = first.evaluate_many(left)
    right_out = second.evaluate_many(right)

    assert not np.allclose(left_out[0][0], right_out[0][0])
    # And each client's own answers stay consistent when asked again.
    first.clear_cache()
    assert np.allclose(first.evaluate_many(left)[0][0], left_out[0][0])


def test_requests_of_different_sizes_are_scattered_correctly(server):
    """Each position gets its own answer back, whatever it was batched with.

    Not bit-identical: a matmul over a batch of 7 picks different kernels than
    one over a batch of 1, so the last couple of digits move. That is inherent to
    batching and applies to the local evaluator too -- what must hold is that the
    answers line up with the requests.
    """
    client = _client(server)
    positions = _planes(7, seed=4)
    one_by_one = [client.evaluate(planes) for planes in positions]
    client.clear_cache()
    together = client.evaluate_many(positions)

    for (logits_a, value_a), (logits_b, value_b) in zip(one_by_one, together):
        assert np.allclose(logits_a, logits_b, atol=1e-4)
        assert value_a == pytest.approx(value_b, abs=1e-4)
        # ...and not the answer meant for a different position.
        assert np.argmax(logits_a) == np.argmax(logits_b)


def test_the_cache_keeps_repeats_off_the_wire(server):
    client = _client(server)
    positions = _planes(4, seed=5)
    client.evaluate_many(positions)
    assert client.misses == 4

    client.evaluate_many(positions)
    assert client.misses == 4, "a second look must be served from the cache"
    assert client.hits == 4


def test_the_server_batches_across_requests(server):
    client = _client(server)
    for index in range(6):
        client.evaluate_many(_planes(8, seed=100 + index))
    assert server.stats.positions == 48
    assert server.stats.batches >= 1
    assert server.stats.positions_per_second > 0


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


def test_a_client_raises_instead_of_hanging_when_nobody_answers():
    context = mp.get_context("spawn")
    orphan = RemoteEvaluator(0, context.Queue(), context.Queue(), timeout=0.3)
    with pytest.raises(RuntimeError, match="stopped answering"):
        orphan.evaluate_many(_planes(1))


def test_a_stale_answer_is_dropped_rather_than_returned(server):
    """A late reply to an abandoned request must not become the next answer."""
    client = _client(server)
    client.responses.put((-1, np.zeros((1, 4672), np.float32), np.zeros(1, np.float32)))
    logits, value = client.evaluate(_planes(1, seed=9)[0])
    assert not np.array_equal(logits, np.zeros(4672, np.float32))


def test_stopping_twice_is_harmless(server):
    first = server.stop()
    assert server.stop().positions == first.positions


def test_no_gpu_means_no_server_rather_than_a_crash(checkpoint):
    config = RunConfig()
    config.inference.enabled = True
    config.inference.device = "cpu"
    context = mp.get_context("spawn")
    assert _start_inference_server(checkpoint, config, 2, context) is None


def test_disabled_by_default(checkpoint):
    context = mp.get_context("spawn")
    assert _start_inference_server(checkpoint, RunConfig(), 2, context) is None


def test_presets_leave_the_shared_gpu_off_until_it_earns_its_place():
    """Measured at 0.84x-1.55x depending on shape; not a default yet.

    See the table in ``app.engine.inference``. The knobs are in the presets so
    the experiment is one edit away, but nobody gets it by surprise.
    """
    for name in ("tiny", "small", "medium", "large", "policy-only"):
        config = preset_config(name)
        assert config.inference.enabled is False
        assert config.inference.max_batch >= 128


# --------------------------------------------------------------------------- #
# Through a real game
# --------------------------------------------------------------------------- #


def test_a_game_played_remotely_is_the_same_game(server, checkpoint):
    """The evaluator moved processes; the moves must not move with it."""
    config = RunConfig()
    config.network = NETWORK
    config.search.simulations = 8
    config.selfplay.max_game_plies = 12
    config.selfplay.resign_threshold = None

    local = play_game(Evaluator(*_load(checkpoint)), config, np.random.default_rng(4))
    remote = play_game(_client(server), config, np.random.default_rng(4))

    assert remote.moves_uci == local.moves_uci
    assert remote.result == local.result
    assert [s.value for s in remote.samples] == [s.value for s in local.samples]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_pool_really_uses_the_shared_gpu(checkpoint):
    config = RunConfig()
    config.network = NETWORK
    config.search.simulations = 8
    config.selfplay.max_game_plies = 10
    config.selfplay.workers = 2
    config.selfplay.resign_threshold = None
    config.inference.enabled = True
    config.inference.device = "cuda"

    batch = generate_selfplay(checkpoint, config, games=2, seed=1)

    assert len(batch.games) == 2
    assert batch.inference is not None
    assert batch.inference["positions"] > 0
    assert batch.inference["batches"] > 0
