"""Classical-engine analysis -- the *human player's* assistance.

Access rules enforced here (see ``docs/08-isolation.md``):

* assistance is tied to a game in which a **human** is one of the players;
* it can be switched off per game (``assistance_enabled``) so a human can play
  the network unaided;
* games without a human (network vs Stockfish benchmark matches) return 403.

The neural engine never calls these endpoints -- it has no HTTP client, and
``app.engine`` does not import the Stockfish service at all.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from app.api.schemas import AnalysisRequest, HintRequest
from app.services.game_manager import GameError, get_game_manager
from app.services.stockfish_service import StockfishUnavailable, get_stockfish

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


def _serialise(result) -> Dict:
    return {
        "fen": result.fen,
        "depth": result.depth,
        "time_ms": result.time_ms,
        "engine": result.engine,
        "lines": [
            {
                "rank": line.rank,
                "score": asdict(line.score),
                "moves_uci": line.moves_uci,
                "moves_san": line.moves_san,
                "depth": line.depth,
                "nodes": line.nodes,
            }
            for line in result.lines
        ],
    }


@router.get("/engine")
async def engine_info() -> Dict:
    return get_stockfish().info()


@router.post("/position")
async def analyse_position(request: AnalysisRequest) -> Dict:
    """Free analysis of an arbitrary FEN (analysis board, not tied to a game)."""
    stockfish = get_stockfish()
    try:
        result = await run_in_threadpool(
            stockfish.analyse,
            request.fen,
            depth=request.depth,
            multipv=request.multipv,
            movetime_ms=request.movetime_ms,
        )
    except StockfishUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=f"invalid FEN: {error}") from error
    return _serialise(result)


@router.post("/hint")
async def hint(request: HintRequest) -> Dict:
    """Evaluation + best moves for the human side of an ongoing game."""
    try:
        session = get_game_manager().get(request.game_id)
    except GameError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    if session.human_color is None:
        raise HTTPException(
            status_code=403,
            detail="assistance is only available in games with a human player",
        )
    if not session.assistance_enabled:
        raise HTTPException(
            status_code=403, detail="assistance is disabled for this game"
        )

    stockfish = get_stockfish()
    try:
        result = await run_in_threadpool(
            stockfish.analyse,
            session.board.fen(),
            depth=request.depth,
            multipv=request.multipv,
        )
    except StockfishUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    payload = _serialise(result)
    payload["human_to_move"] = session.human_to_move
    payload["for_color"] = "white" if session.human_color else "black"
    return payload
