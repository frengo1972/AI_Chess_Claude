"""Training control and KPI streaming for the dashboard."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from app.api.schemas import (
    StartTrainingRequest,
    StopTrainingRequest,
    WatchSettingsRequest,
)
from app.config import PRESETS, preset_config
from app.services.training_manager import LaunchRequest, get_training_manager

router = APIRouter(prefix="/api/training", tags=["training"])


@router.get("/presets")
async def presets() -> Dict:
    detail = {}
    for name in PRESETS:
        config = preset_config(name)
        from app.engine.network import ChessNet

        described = ChessNet(config.network).describe()
        detail[name] = {
            "config": config.to_dict(),
            "network": described,
            "summary": {
                "blocks": config.network.residual_blocks,
                "filters": config.network.filters,
                "history": config.network.history_length,
                "simulations": config.search.simulations,
                "games_per_iteration": config.selfplay.games_per_iteration,
                "parameters": described["parameters"],
                "size_mb": described["size_mb"],
            },
        }
    return {"presets": detail, "default": "small"}


@router.get("/runs")
async def runs(limit: int = Query(50, ge=1, le=200)) -> Dict:
    return {"runs": get_training_manager().runs(limit=limit)}


@router.get("/status")
async def status(run_id: Optional[str] = None) -> Dict:
    return get_training_manager().status(run_id)


@router.get("/history")
async def history(run_id: str, limit: int = Query(2000, ge=1, le=100_000)) -> Dict:
    return {"run_id": run_id, "iterations": get_training_manager().history(run_id, limit)}


@router.get("/events")
async def events(run_id: str, limit: int = Query(200, ge=1, le=2000)) -> Dict:
    return {"run_id": run_id, "events": get_training_manager().events(run_id, limit)}


@router.get("/games")
async def sample_games(run_id: str, limit: int = Query(6, ge=1, le=60)) -> Dict:
    manager = get_training_manager()
    return {"run_id": run_id, "games": manager.store.sample_games(run_id, limit)}


@router.post("/start")
async def start(request: StartTrainingRequest) -> Dict:
    manager = get_training_manager()
    try:
        return manager.start(
            LaunchRequest(
                preset=request.preset,
                name=request.name or request.preset,
                iterations=request.iterations,
                games_per_iteration=request.games_per_iteration,
                simulations=request.simulations,
                workers=request.workers,
                device=request.device,
                resume_run_id=request.resume_run_id,
                overrides=request.overrides,
            )
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/stop")
async def stop(request: StopTrainingRequest) -> Dict:
    manager = get_training_manager()
    try:
        return manager.stop(request.run_id, force=request.force)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/watch")
async def watch(run_id: Optional[str] = None) -> Dict:
    """The self-play games in progress, as the workers last published them."""
    return get_training_manager().watch(run_id)


@router.post("/watch/settings")
async def watch_settings(request: WatchSettingsRequest) -> Dict:
    manager = get_training_manager()
    try:
        return manager.set_watch(
            request.run_id,
            enabled=request.enabled,
            move_delay_ms=request.move_delay_ms,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.websocket("/ws")
async def training_socket(socket: WebSocket, run_id: Optional[str] = None) -> None:
    """Push status + last iteration every second while the dashboard is open."""
    await socket.accept()
    manager = get_training_manager()
    try:
        while True:
            payload = manager.status(run_id)
            await socket.send_text(json.dumps(payload, default=str))
            await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        return
    except Exception:  # noqa: BLE001 - a dead socket must not spam the log
        return


@router.websocket("/watch/ws")
async def watch_socket(
    socket: WebSocket,
    run_id: Optional[str] = None,
    interval_ms: int = Query(500, ge=150, le=5_000),
) -> None:
    """Push the live boards on their own socket.

    Separate from ``/ws`` because boards need a faster cadence than KPIs, and
    because nobody should pay for them with the dashboard's watch panel closed.
    """
    await socket.accept()
    manager = get_training_manager()
    interval = interval_ms / 1000.0
    try:
        while True:
            await socket.send_text(json.dumps(manager.watch(run_id), default=str))
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        return
    except Exception:  # noqa: BLE001 - a dead socket must not spam the log
        return
