"""Game endpoints: create a game, move, let the engine reply, undo, resign."""

from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from app.api.schemas import (
    MoveRequest,
    NeuralEvaluationRequest,
    NewGameRequest,
    UndoRequest,
)
from app.services.game_manager import GameError, OpponentSpec, get_game_manager
from app.services.nn_engine import get_neural_engine

router = APIRouter(prefix="/api/game", tags=["game"])


def _session(game_id: str):
    try:
        return get_game_manager().get(game_id)
    except GameError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("")
async def create_game(request: NewGameRequest) -> Dict:
    manager = get_game_manager()
    spec = OpponentSpec(
        kind=request.opponent.kind,
        model_id=request.opponent.model_id,
        simulations=request.opponent.simulations,
        temperature=request.opponent.temperature,
        stockfish_elo=request.opponent.stockfish_elo,
        stockfish_skill=request.opponent.stockfish_skill,
        stockfish_movetime_ms=request.opponent.stockfish_movetime_ms,
    )
    session = manager.create(
        human_color=request.human_color,
        opponent=spec,
        assistance_enabled=request.assistance_enabled,
        starting_fen=request.starting_fen,
    )

    # If the human is Black, the engine opens.
    engine_report = None
    if not session.human_to_move and session.human_color is not None:
        engine_report = await run_in_threadpool(manager.engine_move, session)

    return {"game": session.to_payload(), "engine_move": engine_report}


@router.get("/{game_id}")
async def get_game(game_id: str) -> Dict:
    return {"game": _session(game_id).to_payload()}


@router.post("/{game_id}/move")
async def play_move(game_id: str, request: MoveRequest) -> Dict:
    manager = get_game_manager()
    session = _session(game_id)
    try:
        manager.apply_move(session, request.to_uci())
    except GameError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    engine_report = None
    if not session.is_over and session.human_color is not None:
        engine_report = await run_in_threadpool(manager.engine_move, session)

    return {"game": session.to_payload(), "engine_move": engine_report}


@router.post("/{game_id}/engine-move")
async def engine_move(game_id: str) -> Dict:
    manager = get_game_manager()
    session = _session(game_id)
    try:
        report = await run_in_threadpool(manager.engine_move, session)
    except GameError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"game": session.to_payload(), "engine_move": report}


@router.post("/{game_id}/undo")
async def undo(game_id: str, request: UndoRequest) -> Dict:
    manager = get_game_manager()
    session = _session(game_id)
    manager.undo(session, request.plies)
    return {"game": session.to_payload()}


@router.post("/{game_id}/resign")
async def resign(game_id: str) -> Dict:
    manager = get_game_manager()
    session = _session(game_id)
    manager.resign(session)
    return {"game": session.to_payload()}


@router.delete("/{game_id}")
async def delete_game(game_id: str) -> Dict:
    get_game_manager().delete(game_id)
    return {"deleted": game_id}


@router.post("/{game_id}/neural-evaluation")
async def neural_evaluation(game_id: str, request: NeuralEvaluationRequest) -> Dict:
    """What the *network* thinks of the current position.

    This is the network's own output, not classical-engine help, so it is safe
    to show at any time -- including while the network is to move.
    """
    session = _session(game_id)
    spec = session.opponent
    engine = get_neural_engine()
    model_id = spec.model_id or engine.default_model_id()
    payload = await run_in_threadpool(
        engine.evaluate_position,
        session.moves_uci,
        model_id=model_id,
        simulations=request.simulations,
    )
    return payload


@router.get("/{game_id}/pgn")
async def pgn(game_id: str) -> Dict:
    session = _session(game_id)
    return {"pgn": session.pgn()}
