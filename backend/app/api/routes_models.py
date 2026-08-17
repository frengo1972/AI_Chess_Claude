"""Model registry, system information, and the network-vs-Stockfish benchmark."""

from __future__ import annotations

import platform
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from app.api.schemas import BenchmarkRequest
from app.config import CHECKPOINT_DIR, RunConfig
from app.services.nn_engine import UNTRAINED_ID, get_neural_engine
from app.services.stockfish_service import get_stockfish

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models")
async def list_models() -> Dict:
    engine = get_neural_engine()
    models = [asdict(model) for model in engine.list_models()]
    return {"models": models, "default": engine.default_model_id()}


@router.post("/models/reload")
async def reload_models() -> Dict:
    get_neural_engine().invalidate()
    return {"reloaded": True}


@router.get("/system")
async def system() -> Dict:
    import torch

    engine = get_neural_engine()
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda,
        "cuda_device": torch.cuda.get_device_name(0) if cuda else None,
        "device": engine.device,
        "stockfish": get_stockfish().info(),
        "checkpoint_dir": str(CHECKPOINT_DIR),
    }


@router.post("/benchmark")
async def benchmark(request: BenchmarkRequest) -> Dict:
    """Play the selected network against Stockfish at a capped Elo.

    Pure measurement: the result is a score, and nothing flows back into the
    network. Can take a while -- the caller should expect a long request.
    """
    engine = get_neural_engine()
    if request.model_id == UNTRAINED_ID:
        raise HTTPException(
            status_code=400, detail="the untrained network cannot be benchmarked"
        )
    try:
        _, info = engine.get(request.model_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    if not info.path:
        raise HTTPException(status_code=400, detail="model has no checkpoint on disk")

    config_path = Path(info.path).parent / "config.json"
    config = (
        RunConfig.from_json(config_path)
        if config_path.exists()
        else RunConfig()
    )
    config.network.history_length = info.history_length

    from app.services.benchmark import play_network_vs_stockfish

    try:
        summary = await run_in_threadpool(
            play_network_vs_stockfish,
            Path(info.path),
            config,
            games=request.games,
            stockfish_elo=request.stockfish_elo,
            movetime_ms=request.movetime_ms,
            simulations=request.simulations,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    summary["model_id"] = request.model_id
    return summary
