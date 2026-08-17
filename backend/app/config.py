"""Central configuration: paths, network shape, search and training presets.

Everything the experiment can be tuned with lives here so that scaling from a
two-hour run to a multi-day run is a config change, not a code change.
Presets are exposed to the UI, and a run's exact config is stored next to its
checkpoints so results stay reproducible.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

DATA_DIR = Path(os.environ.get("AICHESS_DATA_DIR", BACKEND_DIR / "data"))
CHECKPOINT_DIR = Path(os.environ.get("AICHESS_CHECKPOINT_DIR", BACKEND_DIR / "checkpoints"))
ENGINES_DIR = Path(os.environ.get("AICHESS_ENGINES_DIR", PROJECT_ROOT / "engines"))
DOCS_DIR = Path(os.environ.get("AICHESS_DOCS_DIR", PROJECT_ROOT / "docs"))

for _directory in (DATA_DIR, CHECKPOINT_DIR, ENGINES_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

TRAINING_DB = DATA_DIR / "training.db"
REPLAY_DIR = DATA_DIR / "replay"


# --------------------------------------------------------------------------- #
# Config dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class NetworkConfig:
    """Shape of the policy/value residual network."""

    history_length: int = 4
    """Number of past positions stacked into the input (T). 8 = AlphaZero."""

    residual_blocks: int = 6
    filters: int = 96
    value_hidden: int = 128
    policy_head_filters: int = 73
    """73 makes the policy head a single 3x3 conv straight onto the move planes."""

    se_ratio: int = 0
    """Squeeze-Excitation reduction ratio inside residual blocks; 0 disables it."""


@dataclass
class SearchConfig:
    """PUCT / MCTS parameters.

    ``simulations = 1`` degenerates to "pure policy network": the search does a
    single evaluation of the root and plays the highest-prior legal move. That
    is the cheap mode; raising ``simulations`` turns the same network into an
    AlphaZero-style player.
    """

    simulations: int = 96
    c_puct: float = 1.6
    c_puct_base: float = 19652.0
    """Set > 0 to use the AlphaZero visit-count-dependent c_puct schedule."""

    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    """Root exploration noise. Applied during self-play only, never when playing
    against a human or a benchmark engine."""

    temperature: float = 1.0
    temperature_moves: int = 20
    """Sample proportionally to visit counts for this many plies, then play the
    most-visited move deterministically."""

    fpu_reduction: float = 0.25
    """First-Play-Urgency: unvisited children start at parent_q - this."""

    virtual_loss: float = 1.0
    max_batch_size: int = 16
    """Leaves collected before a single batched network evaluation."""


@dataclass
class SelfPlayConfig:
    games_per_iteration: int = 40
    workers: int = 6
    max_game_plies: int = 240
    """Games longer than this are scored as draws; prevents shuffling forever."""

    resign_threshold: float = -0.92
    resign_consecutive: int = 6
    resign_disable_fraction: float = 0.1
    """Fraction of games played without resignation, to measure false positives."""

    opening_random_plies: int = 0
    """Optional uniformly-random opening plies for extra diversity."""

    device: str = "cpu"
    """Self-play workers default to CPU: many small-batch processes beat one
    contended GPU, and it avoids CUDA-in-subprocess issues on Windows."""

    torch_threads_per_worker: int = 1

    watch_enabled: bool = False
    """Publish live self-play boards for the dashboard. Off by default because it
    writes a small file per ply; it can be switched on later from the UI without
    restarting the run (see ``app.engine.watch``)."""

    watch_move_delay_ms: int = 0
    """Pause after each self-play move while watching, so a human can follow the
    game. Anything above zero slows training down proportionally: it is an
    observation aid, not a training parameter."""


@dataclass
class InferenceConfig:
    """One GPU serving the whole self-play pool, instead of a model per worker.

    A single worker never has more than ``search.max_batch_size`` leaves in
    flight, which is far too few to fill a GPU; merging every worker's leaves
    into one forward pass is what makes the GPU worth using at all. See
    ``app.engine.inference`` for the measurements behind the defaults.
    """

    enabled: bool = False
    """Off in the dataclass default so a bare ``RunConfig`` keeps the simple
    per-worker path; the presets turn it on."""

    device: str = "auto"
    """``auto`` falls back to per-worker CPU evaluators when there is no GPU."""

    max_batch: int = 256
    collect_timeout_ms: float = 2.0
    """How long the server waits for more workers' requests before running the
    batch. Long enough to gather the pool, short enough not to add latency."""

    use_amp: bool = False
    """fp16 measured as noise on small networks (kernel-launch bound); worth
    revisiting on larger towers."""


@dataclass
class TrainConfig:
    iterations: int = 100
    batch_size: int = 256
    steps_per_iteration: int = 400
    optimizer: str = "adamw"
    """``adamw`` converges faster on short runs; ``sgd`` matches AlphaZero."""

    learning_rate: float = 2e-3
    lr_milestones: tuple = (30, 60, 85)
    lr_gamma: float = 0.1
    weight_decay: float = 1e-4
    momentum: float = 0.9
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    grad_clip: float = 2.0

    value_search_weight: float = 0.0
    """Share of the value target taken from the search value ``q`` at that
    position instead of the final game result ``z``::

        target = (1 - w) * z + w * q

    ``z`` is one very noisy sample: a single outcome pinned onto every position of
    the game, all of them highly correlated. ``q`` is what the tree actually
    concluded about *that* position, so mixing it in cuts the variance of the
    value target sharply -- the network bootstraps off its own search, not off
    human chess knowledge. ``0.0`` is the original AlphaZero target and stays the
    dataclass default for backwards compatibility; the presets ship ``0.5``.

    Ignored when ``search.simulations == 1``: with no search, ``q`` *is* the value
    head's own output, and regressing it onto itself teaches nothing."""

    replay_buffer_size: int = 200_000
    min_buffer_before_training: int = 4_000
    sample_recent_fraction: float = 1.0

    device: str = "auto"
    checkpoint_every: int = 1
    keep_last_checkpoints: int = 10


@dataclass
class ArenaConfig:
    """Gating match between the freshly trained net and the current best."""

    enabled: bool = True
    games: int = 20
    every_iterations: int = 2
    simulations: int = 96
    win_threshold: float = 0.55
    """Score (win = 1, draw = 0.5) needed to promote the challenger."""

    max_plies: int = 240


@dataclass
class BenchmarkConfig:
    """Periodic calibration against Stockfish at a limited strength.

    This measures the network; it never feeds it. See ``docs/08-isolation.md``.
    """

    enabled: bool = False
    every_iterations: int = 10
    games: int = 10
    stockfish_elo: int = 1350
    movetime_ms: int = 50
    simulations: int = 96


@dataclass
class RunConfig:
    name: str = "run"
    seed: int = 0
    network: NetworkConfig = field(default_factory=NetworkConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    # -- serialisation ----------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RunConfig":
        base = cls()
        return replace(
            base,
            name=payload.get("name", base.name),
            seed=payload.get("seed", base.seed),
            network=_merge(NetworkConfig, base.network, payload.get("network")),
            search=_merge(SearchConfig, base.search, payload.get("search")),
            selfplay=_merge(SelfPlayConfig, base.selfplay, payload.get("selfplay")),
            inference=_merge(InferenceConfig, base.inference, payload.get("inference")),
            train=_merge(TrainConfig, base.train, payload.get("train")),
            arena=_merge(ArenaConfig, base.arena, payload.get("arena")),
            benchmark=_merge(BenchmarkConfig, base.benchmark, payload.get("benchmark")),
        )

    @classmethod
    def from_json(cls, path: Path) -> "RunConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _merge(dataclass_type, current, overrides: Optional[Dict[str, Any]]):
    if not overrides:
        return current
    known = {f for f in dataclass_type.__dataclass_fields__}
    clean = {k: v for k, v in overrides.items() if k in known}
    return replace(current, **clean)


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #

PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny": {
        "name": "tiny",
        "inference": {"enabled": False, "max_batch": 128},
        "network": {"history_length": 2, "residual_blocks": 3, "filters": 64},
        "search": {"simulations": 32},
        "selfplay": {"games_per_iteration": 20, "workers": 6, "max_game_plies": 160},
        "train": {"steps_per_iteration": 150, "batch_size": 128,
                  "min_buffer_before_training": 1500, "replay_buffer_size": 60_000,
                  "value_search_weight": 0.5},
        "arena": {"games": 12, "simulations": 32},
    },
    "small": {
        "name": "small",
        "inference": {"enabled": False, "max_batch": 256},
        "network": {"history_length": 4, "residual_blocks": 6, "filters": 96},
        "search": {"simulations": 96},
        "selfplay": {"games_per_iteration": 40, "workers": 8, "max_game_plies": 240},
        "train": {"steps_per_iteration": 400, "batch_size": 256,
                  "min_buffer_before_training": 4_000, "replay_buffer_size": 200_000,
                  "value_search_weight": 0.5},
        "arena": {"games": 20, "simulations": 96},
    },
    "medium": {
        "name": "medium",
        "inference": {"enabled": False, "max_batch": 512},
        "network": {"history_length": 8, "residual_blocks": 10, "filters": 128},
        "search": {"simulations": 200},
        "selfplay": {"games_per_iteration": 80, "workers": 10, "max_game_plies": 300},
        "train": {"steps_per_iteration": 800, "batch_size": 512,
                  "min_buffer_before_training": 20_000, "replay_buffer_size": 600_000,
                  "value_search_weight": 0.5},
        "arena": {"games": 30, "simulations": 200},
    },
    "large": {
        "name": "large",
        "inference": {"enabled": False, "max_batch": 512},
        "network": {"history_length": 8, "residual_blocks": 20, "filters": 256},
        "search": {"simulations": 400},
        "selfplay": {"games_per_iteration": 160, "workers": 12, "max_game_plies": 340},
        "train": {"steps_per_iteration": 1500, "batch_size": 512,
                  "min_buffer_before_training": 50_000, "replay_buffer_size": 1_500_000,
                  "value_search_weight": 0.5},
        "arena": {"games": 40, "simulations": 400},
    },
    "policy-only": {
        "name": "policy-only",
        "inference": {"enabled": False, "max_batch": 512},
        "network": {"history_length": 4, "residual_blocks": 6, "filters": 96},
        "search": {"simulations": 1, "dirichlet_epsilon": 0.35, "temperature_moves": 30},
        "selfplay": {"games_per_iteration": 400, "workers": 10, "max_game_plies": 200},
        # No search means no independent q to mix in: the target stays pure z.
        "train": {"steps_per_iteration": 600, "batch_size": 256,
                  "min_buffer_before_training": 8_000, "replay_buffer_size": 300_000,
                  "value_search_weight": 0.0},
        "arena": {"games": 60, "simulations": 1},
    },
}

DEFAULT_PRESET = "small"


def preset_config(name: str = DEFAULT_PRESET) -> RunConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return RunConfig.from_dict(PRESETS[name])


# --------------------------------------------------------------------------- #
# Server settings
# --------------------------------------------------------------------------- #


@dataclass
class ServerSettings:
    host: str = os.environ.get("AICHESS_HOST", "127.0.0.1")
    port: int = int(os.environ.get("AICHESS_PORT", "8077"))
    """8077 rather than 8000: port 8000 is a busy default on dev machines."""

    cors_origins: tuple = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )
    stockfish_path: Optional[str] = os.environ.get("AICHESS_STOCKFISH_PATH")
    max_concurrent_games: int = 32
    game_ttl_seconds: int = 60 * 60 * 6


SERVER = ServerSettings()
