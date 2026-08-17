"""FastAPI application entry point.

Run it with::

    uvicorn app.main:app --reload --port 8000

The API is split by concern, and the split matters: ``routes_game`` drives the
neural engine, ``routes_analysis`` drives Stockfish for the human, and the two
never share a code path.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import (
    routes_analysis,
    routes_docs,
    routes_game,
    routes_models,
    routes_training,
)
from app.config import SERVER
from app.services.stockfish_service import get_stockfish

logger = logging.getLogger("aichess")


@asynccontextmanager
async def lifespan(app: FastAPI):
    stockfish = get_stockfish()
    logger.info("stockfish: %s", stockfish.info())
    yield
    stockfish.close()


app = FastAPI(
    title="AI_Chess_Claude",
    version=__version__,
    description=(
        "Self-play reinforcement-learning chess engine, with a classical engine "
        "available to the human player only."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SERVER.cors_origins),
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_game.router)
app.include_router(routes_analysis.router)
app.include_router(routes_training.router)
app.include_router(routes_models.router)
app.include_router(routes_docs.router)


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": __version__})


def main() -> None:
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=SERVER.host,
        port=SERVER.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
