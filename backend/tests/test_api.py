"""End-to-end HTTP behaviour, including the human-only assistance rule."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.stockfish_service import get_stockfish


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def stockfish_available():
    return get_stockfish().available


def _new_game(client, **overrides):
    payload = {
        "human_color": "white",
        "opponent": {"kind": "neural", "model_id": "untrained", "simulations": 2},
        "assistance_enabled": True,
    }
    payload.update(overrides)
    response = client.post("/api/game", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_health_and_system(client):
    assert client.get("/api/health").json()["status"] == "ok"
    system = client.get("/api/system").json()
    assert "torch" in system
    assert "stockfish" in system


def test_models_endpoint_always_offers_a_playable_network(client):
    payload = client.get("/api/models").json()
    ids = {model["id"] for model in payload["models"]}
    assert "untrained" in ids
    assert payload["default"] in ids


def test_new_game_returns_legal_moves(client):
    game = _new_game(client)["game"]
    assert game["turn"] == "white"
    assert game["human_to_move"] is True
    assert game["legal_moves"]["e2"] == ["e3", "e4"]
    assert len(game["legal_moves"]) == 10  # 8 pawns + 2 knights


def test_human_move_triggers_an_engine_reply(client):
    game_id = _new_game(client)["game"]["id"]
    response = client.post(
        f"/api/game/{game_id}/move", json={"from": "e2", "to": "e4"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["game"]["moves_san"][0] == "e4"
    assert body["engine_move"] is not None
    assert body["engine_move"]["source"] == "neural"
    assert len(body["game"]["moves_uci"]) == 2
    assert body["game"]["turn"] == "white"


def test_illegal_move_is_rejected(client):
    game_id = _new_game(client)["game"]["id"]
    response = client.post(
        f"/api/game/{game_id}/move", json={"from": "e2", "to": "e5"}
    )
    assert response.status_code == 400
    assert "illegal" in response.json()["detail"]


def test_undo_and_resign(client):
    game_id = _new_game(client)["game"]["id"]
    client.post(f"/api/game/{game_id}/move", json={"from": "d2", "to": "d4"})
    undone = client.post(f"/api/game/{game_id}/undo", json={"plies": 2}).json()
    assert undone["game"]["moves_uci"] == []

    resigned = client.post(f"/api/game/{game_id}/resign").json()
    assert resigned["game"]["is_over"] is True
    assert resigned["game"]["termination"] == "resignation"


def test_playing_black_lets_the_engine_open(client):
    body = _new_game(client, human_color="black")
    assert body["engine_move"] is not None
    assert body["game"]["turn"] == "black"
    assert body["game"]["human_to_move"] is True


def test_neural_evaluation_reports_the_networks_own_view(client):
    game_id = _new_game(client)["game"]["id"]
    response = client.post(
        f"/api/game/{game_id}/neural-evaluation",
        json={"game_id": game_id, "simulations": 4},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert -1.0 <= payload["value"] <= 1.0
    assert payload["top_moves"]


# --------------------------------------------------------------------------- #
# The human-only assistance rule
# --------------------------------------------------------------------------- #


def test_hint_is_refused_when_no_human_is_playing(client):
    """A network-vs-engine game must not be able to obtain Stockfish help."""
    game_id = _new_game(client, human_color="none")["game"]["id"]
    response = client.post("/api/analysis/hint", json={"game_id": game_id, "depth": 6})
    assert response.status_code == 403
    assert "human" in response.json()["detail"]


def test_hint_is_refused_when_assistance_is_disabled(client):
    game_id = _new_game(client, assistance_enabled=False)["game"]["id"]
    response = client.post("/api/analysis/hint", json={"game_id": game_id, "depth": 6})
    assert response.status_code == 403
    assert "disabled" in response.json()["detail"]


def test_hint_returns_lines_for_a_human_game(client, stockfish_available):
    if not stockfish_available:
        pytest.skip("Stockfish binary not installed")
    game_id = _new_game(client)["game"]["id"]
    response = client.post(
        "/api/analysis/hint", json={"game_id": game_id, "depth": 8, "multipv": 3}
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["for_color"] == "white"
    assert 1 <= len(payload["lines"]) <= 3
    assert payload["lines"][0]["moves_san"]


def test_position_analysis_reports_a_mate(client, stockfish_available):
    if not stockfish_available:
        pytest.skip("Stockfish binary not installed")
    response = client.post(
        "/api/analysis/position",
        json={"fen": "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "depth": 10, "multipv": 1},
    )
    assert response.status_code == 200, response.text
    best = response.json()["lines"][0]
    assert best["score"]["mate_in"] == 1
    assert best["moves_san"][0] in {"Ra8#", "Ra8+"}


# --------------------------------------------------------------------------- #
# Training + docs
# --------------------------------------------------------------------------- #


def test_training_presets_expose_network_sizes(client):
    payload = client.get("/api/training/presets").json()
    assert "small" in payload["presets"]
    summary = payload["presets"]["small"]["summary"]
    assert summary["parameters"] > 0
    assert summary["simulations"] >= 1
    assert payload["presets"]["policy-only"]["summary"]["simulations"] == 1


def test_training_status_is_safe_with_no_active_run(client):
    payload = client.get("/api/training/status").json()
    assert "active" in payload


def test_watch_endpoint_is_safe_with_no_active_run(client):
    payload = client.get("/api/training/watch").json()
    assert payload["settings"]["enabled"] in (True, False)
    assert isinstance(payload["boards"], list)


def test_watch_settings_are_rejected_when_out_of_range(client):
    response = client.post(
        "/api/training/watch/settings", json={"move_delay_ms": 999_999}
    )
    assert response.status_code == 422


def test_docs_index_and_document(client):
    documents = client.get("/api/docs").json()["documents"]
    assert documents, "documentation should ship with the project"
    slug = documents[0]["slug"]
    body = client.get(f"/api/docs/{slug}").json()
    assert body["markdown"].startswith("#")
    assert client.get("/api/docs/../secrets").status_code in (400, 404)
