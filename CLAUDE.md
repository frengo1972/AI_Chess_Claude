# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AlphaZero-style self-play RL chess: a PyTorch policy/value ResNet learns purely from
self-play, plus a FastAPI backend and a Vite/React UI to play against it and watch
training KPIs. Stockfish is present but strictly for the *human* player (analysis, hints)
and as a measurement yardstick.

## Commands

```powershell
# one-shot setup (venv + torch + npm + Stockfish/piece assets)
.\scripts\setup.ps1            # -Cpu for the CPU torch build
.\scripts\dev.ps1              # API :8077 + Vite :5173 in two windows; -Port, -NoFrontend
```

```bash
# backend (from backend/, with .venv active)
python -m uvicorn app.main:app --port 8077 --reload
python -m pytest -q                                   # full suite
python -m pytest tests/test_engine_isolation.py -q    # single file
python -m pytest tests/test_api.py::test_health_and_system -q   # single test
python -m app.engine.train --preset small --name run1 --iterations 60   # train from CLI

# frontend (from frontend/)
npm run dev      # :5173, proxies /api -> 127.0.0.1:8077
npm run build    # tsc -b && vite build
npm run lint     # oxlint
```

There is no linter/formatter configured for Python.

## The invariant that shapes the code

The network must never see classical-engine output. This is enforced, not conventional:

* Nothing under `backend/app/engine/` imports `app.services`, `chess.engine`, or anything
  `stockfish*`. [test_engine_isolation.py](backend/tests/test_engine_isolation.py) checks
  this statically (AST) **and** at runtime (fresh interpreter, `sys.modules` inspection).
* `train.py` may benchmark against Stockfish, but only via a lazy import inside
  `_run_benchmark` — the test asserts that is the *only* function in the module importing
  `app.services`.
* `NeuralEngineService.choose_move` must not gain a `stockfish`/`engine`/`assist`/`hint`
  parameter; a test asserts on its signature.
* [routes_analysis.py](backend/app/api/routes_analysis.py) returns 403 for hints in games
  with no human player, or when `assistance_enabled` is false.

Any change that touches `app/engine/` or the analysis endpoints should be checked against
these tests before anything else. Rationale: [docs/08-isolamento-stockfish.md](docs/08-isolamento-stockfish.md).

## Architecture

### Training loop (`backend/app/engine/`)

One iteration in [train.py](backend/app/engine/train.py): **self-play → replay buffer →
gradient steps → arena gate → KPI row**. The candidate replaces `best.pt` only if it scores
above `arena.win_threshold` against the current champion; a rejected candidate is rolled
back (model reloaded from `best.pt`, optimizer rebuilt) so the next self-play uses the
champion.

Module roles:

| module | responsibility |
|---|---|
| [encoding.py](backend/app/engine/encoding.py) | board↔tensor (`14*T + 7` planes, mover's POV) and move↔policy index (`73*64 = 4672`). The contract between chess rules and the net. |
| [network.py](backend/app/engine/network.py) | ResNet tower + policy/value heads. Illegal moves masked *outside* the net. |
| [evaluator.py](backend/app/engine/evaluator.py) | batched inference + cache keyed on exact encoded planes. Thread-unsafe by design: one per worker. |
| [mcts.py](backend/app/engine/mcts.py) | PUCT with virtual loss, batched leaf collection. `simulations=1` ⇒ pure policy play. |
| [rules.py](backend/app/engine/rules.py) | which terminations count (repetition/50-move are *automatic* here, unlike FIDE). |
| [replay.py](backend/app/engine/replay.py) | bit-packed samples (~30× smaller than raw planes); `.npz` shards per iteration. |
| [selfplay.py](backend/app/engine/selfplay.py) | `ProcessPoolExecutor` workers; workers get a *checkpoint path*, not a live model (Windows `spawn`, no CUDA across processes). |
| [arena.py](backend/app/engine/arena.py) | gating matches + Elo conversion. |

`PositionHistory` (not a bare FEN) is the unit passed around — repetition counters and the
last `T` plies must match what training saw. This is why `GameSession` stores the move list
and `choose_move` replays it.

### Serving (`backend/app/`)

* [config.py](backend/app/config.py) — all tunables. `PRESETS` (`tiny`/`small`/`medium`/
  `large`/`policy-only`) and dataclasses `NetworkConfig`/`SearchConfig`/`SelfPlayConfig`/
  `TrainConfig`/`ArenaConfig`/`BenchmarkConfig`. A run's exact config is written to its
  checkpoint dir. Paths and the port are overridable via `AICHESS_*` env vars.
* [services/nn_engine.py](backend/app/services/nn_engine.py) — checkpoint registry
  (model id = `<run-id>/<stem>`, plus the synthetic `untrained`) and move selection.
  Process-wide singleton via `get_neural_engine()`; call `invalidate()` after new checkpoints.
* [services/game_manager.py](backend/app/services/game_manager.py) — in-memory sessions,
  per-session lock, TTL eviction. Derives legal destinations, SAN, material, PGN so the
  frontend never re-implements rules.
* [services/training_manager.py](backend/app/services/training_manager.py) — training runs
  as a **separate OS process** (`python -m app.engine.train`), not in the web server. The
  two sides talk through the filesystem and SQLite: trainer writes `status.json` and
  iteration rows; stopping is cooperative via a `stop.flag` file.
* [store/metrics.py](backend/app/store/metrics.py) — SQLite (WAL) at
  `backend/data/training.db`: runs, iterations, events, sample games.
* API split by concern and routed under `/api`: `game`, `analysis`, `training`, `models`,
  `docs`. `routes_game` drives the neural engine, `routes_analysis` drives Stockfish; they
  share no code path. Reference: [docs/09-api.md](docs/09-api.md).

Singletons (`get_neural_engine`, `get_game_manager`, `get_training_manager`,
`get_stockfish`) are module-level lazy globals — follow that pattern rather than adding DI.

### Frontend (`frontend/src/`)

React 19 + react-router, no state library and no UI framework. All HTTP goes through
[api/client.ts](frontend/src/api/client.ts) with types in
[api/types.ts](frontend/src/api/types.ts) — keep those in sync with
[api/schemas.py](backend/app/api/schemas.py). Three pages: `PlayPage`, `TrainingPage`,
`DocsPage` (renders `docs/*.md` fetched from `/api/docs`). Styling is plain CSS with
tokens in [styles/tokens.css](frontend/src/styles/tokens.css); theme via
`data-theme` on `<html>`.

## Conventions

* Prose for users (README, `docs/`, UI strings) is **Italian**; code, identifiers, comments
  and docstrings are **English**. Keep it that way.
* Module docstrings explain *why* a design choice was made, not what the code does — match
  that when adding modules.
* `docs/` is user-facing documentation served by the app itself; adding a `NN-slug.md` file
  makes it appear in the UI automatically. Update the README table too.
* Untracked/generated: `backend/data/`, `backend/checkpoints/`, `engines/`,
  `frontend/public/pieces/` (the last two come from `scripts/download_assets.py`).
* Tests degrade gracefully when Stockfish is absent (see the `stockfish_available` fixture)
  — new Stockfish-dependent tests should do the same.
