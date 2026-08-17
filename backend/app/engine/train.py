"""The reinforcement-learning loop.

One iteration is:

1. **Self-play** -- the current *best* network plays N games against itself,
   with Dirichlet noise at the root and temperature sampling in the opening, so
   the data is diverse.
2. **Learn** -- a copy of the network is trained on positions sampled from the
   replay buffer, towards two targets: the MCTS visit distribution (policy) and
   the eventual game result (value).
3. **Gate** -- the candidate plays a match against the current best. It is
   promoted only if it scores above the threshold. This is what makes progress
   monotone.
4. **Record** -- KPIs go into SQLite so the dashboard can plot them.

Run it directly::

    python -m app.engine.train --preset small --name myrun --iterations 40
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from app.config import CHECKPOINT_DIR, PRESETS, RunConfig, preset_config
from app.engine.arena import run_match, score_to_elo
from app.engine.network import ChessNet, load_checkpoint, resolve_device, save_checkpoint
from app.engine.replay import ReplayBuffer
from app.engine.selfplay import game_to_pgn, generate_selfplay
from app.engine.watch import (
    SETTINGS_FILENAME,
    WatchSettings,
    clear_slots,
    watch_directory,
    write_settings,
)
from app.store.metrics import MetricsStore

STOP_FILENAME = "stop.flag"
STATUS_FILENAME = "status.json"


class TrainingRun:
    """Owns one run's directory, checkpoints, replay buffer and KPI stream."""

    def __init__(
        self,
        config: RunConfig,
        *,
        run_id: Optional[str] = None,
        preset: Optional[str] = None,
        store: Optional[MetricsStore] = None,
        resume: bool = False,
    ) -> None:
        self.config = config
        self.run_id = run_id or f"{config.name}-{uuid.uuid4().hex[:8]}"
        self.preset = preset
        self.directory = CHECKPOINT_DIR / self.run_id
        self.replay_dir = self.directory / "replay"
        self.watch_dir = watch_directory(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_watch(resume=resume)

        self.store = store or MetricsStore()
        self.device = resolve_device(config.train.device)
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)

        self.best_path = self.directory / "best.pt"
        self.latest_path = self.directory / "latest.pt"
        self.candidate_path = self.directory / "candidate.pt"

        self.buffer = ReplayBuffer(
            config.train.replay_buffer_size, config.network.history_length
        )
        self.start_iteration = 0
        self.games_total = 0
        self.positions_total = 0
        self.elo = 0.0

        if resume and self.best_path.exists():
            self.model, info = load_checkpoint(self.best_path, device=self.device)
            self.start_iteration = int(info.get("iteration", 0))
            state_file = self.directory / "state.json"
            if state_file.exists():
                state = json.loads(state_file.read_text(encoding="utf-8"))
                self.start_iteration = int(state.get("iteration", self.start_iteration))
                self.games_total = int(state.get("games_total", 0))
                self.positions_total = int(state.get("positions_total", 0))
                self.elo = float(state.get("elo", 0.0))
            self.buffer.load_directory(self.replay_dir, limit_files=20)
        else:
            self.model = ChessNet(config.network).to(self.device)
            save_checkpoint(self.best_path, self.model, iteration=0)
            shutil.copyfile(self.best_path, self.latest_path)

        self.optimizer = self._build_optimizer()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=list(config.train.lr_milestones),
            gamma=config.train.lr_gamma,
        )

        config.to_json(self.directory / "config.json")
        self.store.create_run(
            self.run_id,
            config.name,
            config.to_dict(),
            preset=preset,
            network=self.model.describe(),
        )

    # -- setup ------------------------------------------------------------- #

    def _prepare_watch(self, *, resume: bool) -> None:
        """Seed the spectator settings, without overruling a live choice.

        On a resumed run the dashboard may already have asked for a specific
        pace; the config only provides the initial value.
        """
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        clear_slots(self.watch_dir)
        existing = (self.watch_dir / SETTINGS_FILENAME).exists()
        if not (resume and existing):
            write_settings(
                self.watch_dir,
                WatchSettings(
                    enabled=self.config.selfplay.watch_enabled,
                    move_delay_ms=self.config.selfplay.watch_move_delay_ms,
                ),
            )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        train = self.config.train
        if train.optimizer.lower() == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=train.learning_rate,
                momentum=train.momentum,
                weight_decay=train.weight_decay,
                nesterov=True,
            )
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=train.learning_rate,
            weight_decay=train.weight_decay,
        )

    # -- control ----------------------------------------------------------- #

    @property
    def stop_requested(self) -> bool:
        return (self.directory / STOP_FILENAME).exists()

    def request_stop(self) -> None:
        (self.directory / STOP_FILENAME).write_text("stop", encoding="utf-8")

    def clear_stop(self) -> None:
        flag = self.directory / STOP_FILENAME
        if flag.exists():
            flag.unlink()

    def _write_status(self, payload: Dict) -> None:
        (self.directory / STATUS_FILENAME).write_text(
            json.dumps(payload, default=str), encoding="utf-8"
        )

    # -- main loop --------------------------------------------------------- #

    def run(self, iterations: Optional[int] = None) -> None:
        total = iterations if iterations is not None else self.config.train.iterations
        self.clear_stop()
        self.store.set_status(self.run_id, "running")
        self.store.log(
            self.run_id,
            f"run started on {self.device}; network = {self.model.describe()}",
        )
        try:
            for step in range(total):
                if self.stop_requested:
                    self.store.log(self.run_id, "stop requested; exiting cleanly")
                    break
                iteration = self.start_iteration + step + 1
                self._run_iteration(iteration)
            self.store.set_status(self.run_id, "stopped" if self.stop_requested else "finished")
        except Exception as error:  # noqa: BLE001 - surface the failure in the UI
            self.store.log(self.run_id, f"run failed: {error!r}", level="error")
            self.store.set_status(self.run_id, "failed")
            raise
        finally:
            self.clear_stop()

    def _run_iteration(self, iteration: int) -> None:
        metrics: Dict[str, float] = {}
        self._write_status(
            {"run_id": self.run_id, "iteration": iteration, "phase": "selfplay"}
        )

        # 1 --------------------------------------------------------- self-play
        selfplay_started = time.perf_counter()
        batch = generate_selfplay(
            self.best_path,
            self.config,
            seed=int(self.rng.integers(1 << 30)),
            watch_dir=self.watch_dir,
            iteration=iteration,
            run_id=self.run_id,
        )
        selfplay_seconds = time.perf_counter() - selfplay_started

        samples = [sample for game in batch.games for sample in game.samples]
        self.buffer.extend(samples)
        self.games_total += len(batch.games)
        self.positions_total += len(samples)

        shard = self.replay_dir / f"iter-{iteration:05d}.npz"
        shard_buffer = ReplayBuffer(len(samples) or 1, self.config.network.history_length)
        shard_buffer.extend(samples)
        shard_buffer.save(shard)
        self._prune_replay_shards()

        summary = batch.summary()
        metrics.update(
            {
                "games_total": self.games_total,
                "positions_total": self.positions_total,
                "buffer_size": len(self.buffer),
                "avg_plies": summary.get("avg_plies", 0.0),
                "draw_rate": summary.get("draw_rate", 0.0),
                "white_win_rate": summary.get("white_wins", 0) / max(len(batch.games), 1),
                "black_win_rate": summary.get("black_wins", 0) / max(len(batch.games), 1),
                "resign_rate": summary.get("resignations", 0) / max(len(batch.games), 1),
                "policy_entropy": summary.get("avg_policy_entropy", 0.0),
                "avg_root_value": summary.get("avg_root_value", 0.0),
                "selfplay_seconds": selfplay_seconds,
                "games_per_minute": summary.get("games_per_minute", 0.0),
                "positions_per_second": len(samples) / max(selfplay_seconds, 1e-6),
                "games_this_iteration": len(batch.games),
            }
        )

        for game in batch.games[:2]:
            self.store.add_sample_game(
                self.run_id,
                iteration,
                result=game.result,
                plies=game.plies,
                termination=game.termination,
                pgn=game_to_pgn(game),
            )

        # 2 ------------------------------------------------------------ learn
        self._write_status(
            {"run_id": self.run_id, "iteration": iteration, "phase": "train"}
        )
        if len(self.buffer) < self.config.train.min_buffer_before_training:
            self.store.log(
                self.run_id,
                f"iteration {iteration}: buffer {len(self.buffer)} < "
                f"{self.config.train.min_buffer_before_training}, collecting more data",
            )
            metrics.update({"policy_loss": None, "value_loss": None, "total_loss": None})
            metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
            metrics["elo"] = self.elo
            metrics["train_seconds"] = 0.0
            self.store.record_iteration(self.run_id, iteration, metrics)
            return

        train_started = time.perf_counter()
        losses = self._train_steps()
        train_seconds = time.perf_counter() - train_started
        self.scheduler.step()
        metrics.update(losses)
        metrics["train_seconds"] = train_seconds
        metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        save_checkpoint(
            self.candidate_path,
            self.model,
            iteration=iteration,
            metadata={"games_total": self.games_total, "elo": self.elo},
        )
        save_checkpoint(
            self.latest_path,
            self.model,
            iteration=iteration,
            metadata={"games_total": self.games_total, "elo": self.elo},
        )

        # 3 ------------------------------------------------------------- gate
        promoted = False
        arena_config = self.config.arena
        should_gate = (
            arena_config.enabled and iteration % max(arena_config.every_iterations, 1) == 0
        )
        if should_gate:
            self._write_status(
                {"run_id": self.run_id, "iteration": iteration, "phase": "arena"}
            )
            match = run_match(
                self.candidate_path,
                self.best_path,
                self.config,
                seed=int(self.rng.integers(1 << 30)),
            )
            metrics["arena_score"] = match.score
            metrics["elo_delta"] = match.elo_difference
            metrics.update(
                {f"arena_{k}": v for k, v in match.summary().items() if k != "score"}
            )
            if match.score >= arena_config.win_threshold:
                promoted = True
                self.elo += match.elo_difference
                shutil.copyfile(self.candidate_path, self.best_path)
                self.store.log(
                    self.run_id,
                    f"iteration {iteration}: promoted candidate "
                    f"(score {match.score:.3f}, {match.elo_difference:+.0f} Elo)",
                )
            else:
                self.store.log(
                    self.run_id,
                    f"iteration {iteration}: candidate rejected (score {match.score:.3f})",
                )
                # Roll back to the champion so the next self-play uses the best net.
                self.model, _ = load_checkpoint(self.best_path, device=self.device)
                self.model.train()
                self.optimizer = self._build_optimizer()
        else:
            # No gate this iteration: the freshly trained net becomes the player.
            shutil.copyfile(self.candidate_path, self.best_path)
            promoted = True

        metrics["promoted"] = int(promoted)
        metrics["elo"] = self.elo

        # 4 -------------------------------------------------------- benchmark
        if (
            self.config.benchmark.enabled
            and iteration % max(self.config.benchmark.every_iterations, 1) == 0
        ):
            self._write_status(
                {"run_id": self.run_id, "iteration": iteration, "phase": "benchmark"}
            )
            benchmark = self._run_benchmark()
            if benchmark:
                metrics.update(benchmark)

        # 5 ----------------------------------------------------------- record
        if iteration % max(self.config.train.checkpoint_every, 1) == 0:
            generation = self.directory / f"gen-{iteration:05d}.pt"
            shutil.copyfile(self.best_path, generation)
            self._prune_generations()

        (self.directory / "state.json").write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "iteration": iteration,
                    "games_total": self.games_total,
                    "positions_total": self.positions_total,
                    "elo": self.elo,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        self.store.record_iteration(self.run_id, iteration, metrics)
        self._write_status(
            {
                "run_id": self.run_id,
                "iteration": iteration,
                "phase": "idle",
                "metrics": metrics,
            }
        )

    # -- pieces ------------------------------------------------------------ #

    def _train_steps(self) -> Dict[str, float]:
        train = self.config.train
        self.model.train()
        policy_losses: List[float] = []
        value_losses: List[float] = []

        for _ in range(train.steps_per_iteration):
            planes, policies, values = self.buffer.sample_batch(
                train.batch_size, self.rng, train.sample_recent_fraction
            )
            planes_t = torch.from_numpy(planes).to(self.device)
            policy_t = torch.from_numpy(policies).to(self.device)
            value_t = torch.from_numpy(values).to(self.device)

            logits, predicted = self.model(planes_t)
            log_probabilities = F.log_softmax(logits, dim=1)
            policy_loss = -(policy_t * log_probabilities).sum(dim=1).mean()
            value_loss = F.mse_loss(predicted, value_t)
            loss = (
                train.policy_loss_weight * policy_loss
                + train.value_loss_weight * value_loss
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), train.grad_clip)
            self.optimizer.step()

            policy_losses.append(float(policy_loss.detach()))
            value_losses.append(float(value_loss.detach()))

        self.model.eval()
        policy_mean = float(np.mean(policy_losses))
        value_mean = float(np.mean(value_losses))
        return {
            "policy_loss": policy_mean,
            "value_loss": value_mean,
            "total_loss": policy_mean + value_mean,
        }

    def _run_benchmark(self) -> Optional[Dict[str, float]]:
        """Play the best net against Stockfish at a fixed strength, for measurement.

        This is a *read-out*, never a teacher: no Stockfish output ever reaches
        the replay buffer. Imported lazily so the engine package keeps no
        static dependency on it.
        """
        try:
            from app.services.benchmark import benchmark_against_stockfish
        except Exception as error:  # noqa: BLE001
            self.store.log(self.run_id, f"benchmark unavailable: {error!r}", "warn")
            return None
        try:
            return benchmark_against_stockfish(
                self.best_path, self.config, self.config.benchmark
            )
        except Exception as error:  # noqa: BLE001
            self.store.log(self.run_id, f"benchmark failed: {error!r}", "warn")
            return None

    def _prune_generations(self) -> None:
        keep = max(self.config.train.keep_last_checkpoints, 1)
        generations = sorted(self.directory.glob("gen-*.pt"))
        for stale in generations[:-keep]:
            stale.unlink(missing_ok=True)

    def _prune_replay_shards(self, keep: int = 25) -> None:
        shards = sorted(self.replay_dir.glob("iter-*.npz"))
        for stale in shards[:-keep]:
            stale.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_config(args: argparse.Namespace) -> RunConfig:
    if args.config:
        config = RunConfig.from_json(Path(args.config))
    else:
        config = preset_config(args.preset)
    if args.name:
        config.name = args.name
    if args.iterations:
        config.train.iterations = args.iterations
    if args.games:
        config.selfplay.games_per_iteration = args.games
    if args.simulations:
        config.search.simulations = args.simulations
        config.arena.simulations = args.simulations
    if args.workers:
        config.selfplay.workers = args.workers
    if args.device:
        config.train.device = args.device
    return config


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Self-play reinforcement learning")
    parser.add_argument("--preset", default="small", choices=sorted(PRESETS))
    parser.add_argument("--config", help="path to a saved config.json (overrides preset)")
    parser.add_argument("--name")
    parser.add_argument("--run-id", help="reuse an existing run directory")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--games", type=int, help="self-play games per iteration")
    parser.add_argument("--simulations", type=int, help="MCTS simulations per move")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", help="cpu | cuda | auto")
    args = parser.parse_args(argv)

    config = build_config(args)
    run = TrainingRun(
        config, run_id=args.run_id, preset=args.preset, resume=args.resume
    )
    print(f"run id : {run.run_id}")
    print(f"device : {run.device}")
    print(f"network: {json.dumps(run.model.describe())}")
    run.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
