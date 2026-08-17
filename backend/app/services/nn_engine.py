"""Serving side of the neural engine: checkpoint registry + move selection.

This is what the *game* endpoints use. It deliberately has no access to
Stockfish: when the network is asked for a move it sees the position, its own
evaluation, and nothing else.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np

from app.config import CHECKPOINT_DIR, NetworkConfig, RunConfig, SearchConfig, preset_config
from app.engine.encoding import PositionHistory
from app.engine.evaluator import Evaluator
from app.engine.mcts import MCTS, SearchResult
from app.engine.network import ChessNet, load_checkpoint, resolve_device

UNTRAINED_ID = "untrained"


@dataclass
class ModelInfo:
    id: str
    label: str
    run_id: Optional[str]
    path: Optional[str]
    iteration: int
    parameters: int
    size_mb: float
    history_length: int
    residual_blocks: int
    filters: int
    elo: Optional[float] = None
    games_total: Optional[int] = None
    trained: bool = True


@dataclass
class EngineMove:
    move_uci: str
    move_san: str
    evaluation: float
    """Root value in [-1, 1] from the moving side's perspective."""

    simulations: int
    nodes: int
    think_ms: int
    top_moves: List[Dict]
    principal_variation: List[str]
    model_id: str


class NeuralEngineService:
    """Loads checkpoints on demand and plays moves with them."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or resolve_device("auto")
        self._evaluators: Dict[str, Tuple[Evaluator, ModelInfo]] = {}
        self._lock = threading.Lock()

    # -- registry ---------------------------------------------------------- #

    def list_models(self) -> List[ModelInfo]:
        models: List[ModelInfo] = []
        for run_directory in sorted(CHECKPOINT_DIR.iterdir()) if CHECKPOINT_DIR.exists() else []:
            if not run_directory.is_dir():
                continue
            for checkpoint in sorted(run_directory.glob("*.pt")):
                # "candidate" is a transient artefact of the gating step.
                if checkpoint.stem == "candidate":
                    continue
                info = self._describe(checkpoint, run_directory.name)
                if info:
                    models.append(info)
        models.append(self._untrained_info())
        return models

    def _describe(self, path: Path, run_id: str) -> Optional[ModelInfo]:
        """Read a checkpoint's shape without instantiating the network.

        ``list_models`` can face a dozen generations of a 90 M-parameter net;
        summing the state dict is far cheaper than building each one.
        """
        try:
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001 - a half-written checkpoint must not 500
            return None
        network_config = NetworkConfig(**payload["network_config"])
        state_dict = payload["state_dict"]
        parameters = sum(tensor.numel() for tensor in state_dict.values())
        size_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in state_dict.values()
        )
        metadata = payload.get("metadata", {}) or {}
        state = _run_state(path.parent)
        return ModelInfo(
            id=f"{run_id}/{path.stem}",
            label=f"{run_id} · {path.stem}",
            run_id=run_id,
            path=str(path),
            iteration=int(payload.get("iteration", 0)),
            parameters=parameters,
            size_mb=round(size_bytes / (1024 * 1024), 3),
            history_length=network_config.history_length,
            residual_blocks=network_config.residual_blocks,
            filters=network_config.filters,
            elo=state.get("elo", metadata.get("elo")),
            games_total=state.get("games_total", metadata.get("games_total")),
        )

    def _untrained_info(self) -> ModelInfo:
        config = preset_config("small").network
        model = ChessNet(config)
        described = model.describe()
        return ModelInfo(
            id=UNTRAINED_ID,
            label="Untrained network (random weights)",
            run_id=None,
            path=None,
            iteration=0,
            parameters=described["parameters"],
            size_mb=described["size_mb"],
            history_length=config.history_length,
            residual_blocks=config.residual_blocks,
            filters=config.filters,
            trained=False,
        )

    def default_model_id(self) -> str:
        models = [m for m in self.list_models() if m.trained]
        if not models:
            return UNTRAINED_ID
        best = [m for m in models if m.id.endswith("/best")]
        pool = best or models
        return max(pool, key=lambda m: (m.iteration, m.games_total or 0)).id

    # -- loading ----------------------------------------------------------- #

    def _resolve_path(self, model_id: str) -> Optional[Path]:
        if model_id == UNTRAINED_ID:
            return None
        run_id, _, stem = model_id.partition("/")
        candidate = CHECKPOINT_DIR / run_id / f"{stem or 'best'}.pt"
        if not candidate.exists():
            raise FileNotFoundError(f"unknown model {model_id!r}")
        return candidate

    def get(self, model_id: str) -> Tuple[Evaluator, ModelInfo]:
        with self._lock:
            cached = self._evaluators.get(model_id)
            if cached is not None:
                return cached

            path = self._resolve_path(model_id)
            if path is None:
                info = self._untrained_info()
                model = ChessNet(preset_config("small").network)
                evaluator = Evaluator(model, self.device)
            else:
                model, _ = load_checkpoint(path, device=self.device)
                info = self._describe(path, path.parent.name) or self._untrained_info()
                evaluator = Evaluator(model, self.device)

            if len(self._evaluators) > 4:
                self._evaluators.clear()
            self._evaluators[model_id] = (evaluator, info)
            return evaluator, info

    def invalidate(self) -> None:
        with self._lock:
            self._evaluators.clear()

    # -- play -------------------------------------------------------------- #

    def choose_move(
        self,
        moves_uci: List[str],
        *,
        model_id: str,
        simulations: int = 96,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        starting_fen: Optional[str] = None,
    ) -> Optional[EngineMove]:
        """Replay the game, search, and return the network's move.

        The full move list is replayed (rather than a bare FEN) so the history
        planes and the repetition counters are exactly what training saw.
        """
        evaluator, info = self.get(model_id)
        root = chess.Board(starting_fen) if starting_fen else chess.Board()
        state = PositionHistory(root, history_length=info.history_length)
        for uci in moves_uci:
            state.push(chess.Move.from_uci(uci))

        search_config = SearchConfig(
            simulations=max(1, simulations),
            dirichlet_epsilon=0.0,  # no exploration noise outside self-play
            temperature=temperature,
            temperature_moves=10_000 if temperature > 0 else 0,
        )
        searcher = MCTS(evaluator, search_config, np.random.default_rng(seed))

        started = time.perf_counter()
        move, result = searcher.select_move(
            state, add_noise=False, temperature=temperature
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if move is None:
            return None

        return EngineMove(
            move_uci=move.uci(),
            move_san=state.board.san(move),
            evaluation=result.root_value,
            simulations=result.simulations,
            nodes=result.nodes_created,
            think_ms=elapsed_ms,
            top_moves=_top_moves_payload(state.board, result),
            principal_variation=_pv_san(state.board, result),
            model_id=model_id,
        )

    def evaluate_position(
        self, moves_uci: List[str], *, model_id: str, simulations: int = 64
    ) -> Dict:
        """The network's own read of the position (used for the NN stats panel)."""
        evaluator, info = self.get(model_id)
        state = PositionHistory(chess.Board(), history_length=info.history_length)
        for uci in moves_uci:
            state.push(chess.Move.from_uci(uci))
        searcher = MCTS(
            evaluator,
            SearchConfig(simulations=max(1, simulations), dirichlet_epsilon=0.0),
            np.random.default_rng(0),
        )
        result = searcher.search(state)
        return {
            "value": result.root_value,
            "simulations": result.simulations,
            "top_moves": _top_moves_payload(state.board, result),
            "model_id": model_id,
            "terminal": result.terminal.reason if result.terminal else None,
        }


def _top_moves_payload(board: chess.Board, result: SearchResult, limit: int = 5) -> List[Dict]:
    payload = []
    for move, probability, visits in result.top_moves(limit):
        payload.append(
            {
                "uci": move.uci(),
                "san": board.san(move),
                "probability": round(float(probability), 4),
                "visits": int(visits),
            }
        )
    return payload


def _pv_san(board: chess.Board, result: SearchResult, limit: int = 8) -> List[str]:
    replay = board.copy()
    line: List[str] = []
    for move in result.principal_variation[:limit]:
        if move not in replay.legal_moves:
            break
        line.append(replay.san(move))
        replay.push(move)
    return line


def _run_state(directory: Path) -> Dict:
    import json

    state_file = directory / "state.json"
    if not state_file.exists():
        return {}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


_SERVICE: Optional[NeuralEngineService] = None


def get_neural_engine() -> NeuralEngineService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = NeuralEngineService()
    return _SERVICE
