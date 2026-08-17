"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Games
# --------------------------------------------------------------------------- #


class OpponentRequest(BaseModel):
    kind: str = Field("neural", pattern="^(neural|stockfish)$")
    model_id: Optional[str] = None
    simulations: int = Field(96, ge=1, le=4000)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    stockfish_elo: Optional[int] = Field(1500, ge=1320, le=3190)
    stockfish_skill: Optional[int] = Field(None, ge=0, le=20)
    stockfish_movetime_ms: int = Field(200, ge=10, le=10_000)


class NewGameRequest(BaseModel):
    human_color: Optional[str] = Field("white", pattern="^(white|black|none)$")
    opponent: OpponentRequest = Field(default_factory=OpponentRequest)
    assistance_enabled: bool = True
    starting_fen: Optional[str] = None


class MoveRequest(BaseModel):
    from_square: str = Field(..., min_length=2, max_length=2, alias="from")
    to_square: str = Field(..., min_length=2, max_length=2, alias="to")
    promotion: Optional[str] = Field(None, pattern="^[qrbn]$")

    model_config = {"populate_by_name": True}

    def to_uci(self) -> str:
        return f"{self.from_square}{self.to_square}{self.promotion or ''}"


class UndoRequest(BaseModel):
    plies: int = Field(2, ge=1, le=200)


# --------------------------------------------------------------------------- #
# Analysis (human assistance)
# --------------------------------------------------------------------------- #


class AnalysisRequest(BaseModel):
    fen: str
    depth: int = Field(14, ge=1, le=30)
    multipv: int = Field(3, ge=1, le=5)
    movetime_ms: Optional[int] = Field(None, ge=20, le=10_000)


class HintRequest(BaseModel):
    game_id: str
    depth: int = Field(14, ge=1, le=30)
    multipv: int = Field(3, ge=1, le=5)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


class StartTrainingRequest(BaseModel):
    preset: str = "small"
    name: Optional[str] = None
    iterations: Optional[int] = Field(None, ge=1, le=100_000)
    games_per_iteration: Optional[int] = Field(None, ge=1, le=100_000)
    simulations: Optional[int] = Field(None, ge=1, le=4000)
    workers: Optional[int] = Field(None, ge=1, le=64)
    device: Optional[str] = Field(None, pattern="^(auto|cpu|cuda)$")
    resume_run_id: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = None


class StopTrainingRequest(BaseModel):
    run_id: Optional[str] = None
    force: bool = False


class WatchSettingsRequest(BaseModel):
    """Toggle the live self-play feed, and how slowly it plays.

    Both fields are optional so the UI can change one without knowing the other.
    """

    run_id: Optional[str] = None
    enabled: Optional[bool] = None
    move_delay_ms: Optional[int] = Field(None, ge=0, le=5_000)


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #


class BenchmarkRequest(BaseModel):
    model_id: str
    games: int = Field(10, ge=2, le=200)
    stockfish_elo: int = Field(1350, ge=1320, le=3190)
    movetime_ms: int = Field(50, ge=10, le=5_000)
    simulations: Optional[int] = Field(None, ge=1, le=4000)


class NeuralEvaluationRequest(BaseModel):
    game_id: str
    simulations: int = Field(64, ge=1, le=2000)
