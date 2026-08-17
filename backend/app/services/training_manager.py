"""Supervises the training process on behalf of the API.

Training runs in its own OS process, not in the web server: it saturates the
CPU with self-play workers and the GPU with backprop, and it must survive a
reload of the API. The two sides communicate through the filesystem and SQLite:

* the API starts/stops the process and reads KPIs from the store,
* the trainer writes ``status.json`` and iteration rows,
* stopping is cooperative -- the API drops a ``stop.flag`` and the trainer
  finishes its current iteration before exiting.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import BACKEND_DIR, CHECKPOINT_DIR, PRESETS, RunConfig, preset_config
from app.engine.train import STATUS_FILENAME, STOP_FILENAME
from app.engine.watch import (
    WatchSettings,
    read_settings,
    read_slots,
    watch_directory,
    write_settings,
)
from app.store.metrics import MetricsStore

LOG_DIRNAME = "logs"


@dataclass
class LaunchRequest:
    preset: str = "small"
    name: Optional[str] = None
    iterations: Optional[int] = None
    games_per_iteration: Optional[int] = None
    simulations: Optional[int] = None
    workers: Optional[int] = None
    device: Optional[str] = None
    resume_run_id: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = None

    def build_config(self) -> RunConfig:
        if self.preset not in PRESETS:
            raise ValueError(f"unknown preset {self.preset!r}")
        config = preset_config(self.preset)
        if self.overrides:
            config = RunConfig.from_dict({**config.to_dict(), **self.overrides})
        if self.name:
            config.name = self.name
        if self.iterations:
            config.train.iterations = int(self.iterations)
        if self.games_per_iteration:
            config.selfplay.games_per_iteration = int(self.games_per_iteration)
        if self.simulations:
            config.search.simulations = int(self.simulations)
            config.arena.simulations = int(self.simulations)
        if self.workers:
            config.selfplay.workers = int(self.workers)
        if self.device:
            config.train.device = self.device
        return config


class TrainingManager:
    def __init__(self, store: Optional[MetricsStore] = None) -> None:
        self.store = store or MetricsStore()
        self._process: Optional[subprocess.Popen] = None
        self._run_id: Optional[str] = None
        self._lock = threading.Lock()

    # -- state ------------------------------------------------------------- #

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _reconcile(self) -> None:
        """A finished process leaves the run row saying 'running'; fix that."""
        if self._process is not None and self._process.poll() is not None:
            if self._run_id:
                run = self.store.get_run(self._run_id)
                if run and run["status"] == "running":
                    code = self._process.returncode
                    self.store.set_status(
                        self._run_id, "finished" if code == 0 else "failed"
                    )
            self._process = None

    # -- control ----------------------------------------------------------- #

    def start(self, request: LaunchRequest) -> Dict[str, Any]:
        with self._lock:
            self._reconcile()
            if self.is_running:
                raise RuntimeError(
                    f"a training run is already active ({self._run_id}); stop it first"
                )

            config = request.build_config()
            run_id = request.resume_run_id
            command = [
                sys.executable,
                "-m",
                "app.engine.train",
                "--preset",
                request.preset,
            ]

            if run_id:
                directory = CHECKPOINT_DIR / run_id
                if not directory.exists():
                    raise ValueError(f"unknown run {run_id!r}")
                stop_flag = directory / STOP_FILENAME
                if stop_flag.exists():
                    stop_flag.unlink()
                command += ["--run-id", run_id, "--resume", "--config",
                            str(directory / "config.json")]
            else:
                config_path = self._stage_config(config)
                command += ["--config", str(config_path), "--name", config.name]

            if request.iterations:
                command += ["--iterations", str(request.iterations)]

            log_directory = CHECKPOINT_DIR / LOG_DIRNAME
            log_directory.mkdir(parents=True, exist_ok=True)
            log_path = log_directory / f"train-{int(time.time())}.log"
            log_file = open(log_path, "w", encoding="utf-8", buffering=1)

            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(BACKEND_DIR)
            environment.setdefault("PYTHONUNBUFFERED", "1")

            creation_flags = 0
            if sys.platform == "win32":
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

            process = subprocess.Popen(
                command,
                cwd=str(BACKEND_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=environment,
                creationflags=creation_flags,
            )
            self._process = process

            # The trainer prints its run id; poll the store instead of parsing.
            self._run_id = run_id or self._await_new_run(config.name, log_path)
            return {
                "run_id": self._run_id,
                "pid": process.pid,
                "log": str(log_path),
                "command": command,
            }

    def _stage_config(self, config: RunConfig) -> Path:
        staging = CHECKPOINT_DIR / "_pending"
        staging.mkdir(parents=True, exist_ok=True)
        path = staging / f"config-{int(time.time())}.json"
        config.to_json(path)
        return path

    def _await_new_run(self, name: str, log_path: Path, timeout: float = 45.0) -> Optional[str]:
        deadline = time.time() + timeout
        known = {run["id"] for run in self.store.list_runs(limit=100)}
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"trainer exited immediately; see {log_path}"
                )
            for run in self.store.list_runs(limit=20):
                if run["id"] not in known and run["name"] == name:
                    return run["id"]
            time.sleep(0.4)
        return None

    def stop(self, run_id: Optional[str] = None, *, force: bool = False) -> Dict[str, Any]:
        target = run_id or self._run_id
        if not target:
            raise RuntimeError("no active run to stop")
        directory = CHECKPOINT_DIR / target
        directory.mkdir(parents=True, exist_ok=True)
        (directory / STOP_FILENAME).write_text("stop", encoding="utf-8")

        if force and self._process is not None and self._process.poll() is None:
            try:
                if sys.platform == "win32":
                    self._process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self._process.terminate()
            except Exception:  # noqa: BLE001
                pass
        return {"run_id": target, "stopping": True, "forced": force}

    # -- reporting --------------------------------------------------------- #

    def status(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        self._reconcile()
        run = (
            self.store.get_run(run_id)
            if run_id
            else (self.store.get_run(self._run_id) if self._run_id else self.store.latest_run())
        )
        if run is None:
            return {
                "active": False,
                "run": None,
                "phase": "idle",
                "process_alive": self.is_running,
            }

        phase = "idle"
        live: Dict[str, Any] = {}
        status_file = CHECKPOINT_DIR / run["id"] / STATUS_FILENAME
        if status_file.exists():
            try:
                live = json.loads(status_file.read_text(encoding="utf-8"))
                phase = live.get("phase", "idle")
            except Exception:  # noqa: BLE001
                pass

        last = self.store.last_iteration(run["id"])
        return {
            "active": run["status"] == "running",
            "process_alive": self.is_running and self._run_id == run["id"],
            "run": run,
            "phase": phase,
            "live": live,
            "last_iteration": last,
            "stop_requested": (CHECKPOINT_DIR / run["id"] / STOP_FILENAME).exists(),
        }

    # -- watching self-play ------------------------------------------------- #

    def _resolve_run_id(self, run_id: Optional[str] = None) -> Optional[str]:
        if run_id:
            return run_id
        if self._run_id:
            return self._run_id
        latest = self.store.latest_run()
        return latest["id"] if latest else None

    def watch(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """The self-play boards the trainer is currently publishing."""
        target = self._resolve_run_id(run_id)
        if not target:
            return {
                "run_id": None,
                "phase": "idle",
                "active": False,
                "settings": WatchSettings().to_dict(),
                "boards": [],
            }
        directory = watch_directory(CHECKPOINT_DIR / target)
        status = self.status(target)
        return {
            "run_id": target,
            "phase": status.get("phase", "idle"),
            "active": bool(status.get("active")),
            "settings": read_settings(directory).to_dict(),
            "boards": read_slots(directory),
        }

    def set_watch(
        self,
        run_id: Optional[str] = None,
        *,
        enabled: Optional[bool] = None,
        move_delay_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Change what the self-play workers publish, on a run already going.

        The workers poll the settings file, so this takes effect within about
        half a second and needs no restart.
        """
        target = self._resolve_run_id(run_id)
        if not target:
            raise RuntimeError("no run to watch")
        directory = watch_directory(CHECKPOINT_DIR / target)
        current = read_settings(directory)
        updated = WatchSettings.from_dict(
            {
                "enabled": current.enabled if enabled is None else bool(enabled),
                "move_delay_ms": (
                    current.move_delay_ms if move_delay_ms is None else move_delay_ms
                ),
            }
        )
        write_settings(directory, updated)
        return {"run_id": target, "settings": updated.to_dict()}

    def history(self, run_id: str, limit: int = 2000) -> List[Dict[str, Any]]:
        return self.store.iterations(run_id, limit=limit)

    def events(self, run_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        return self.store.events(run_id, limit=limit)

    def runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.store.list_runs(limit=limit)


_MANAGER: Optional[TrainingManager] = None


def get_training_manager() -> TrainingManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = TrainingManager()
    return _MANAGER
