"""The core promise of this project, enforced mechanically.

The neural network must learn and play using nothing but self-play. Stockfish
exists for the *human* (analysis, hints) and as a yardstick (benchmark), and
must never be reachable from the code that chooses the network's moves.

Two checks:

1. **Static** -- the modules that generate and choose moves contain no import
   of the classical engine, directly or via ``app.services``.
2. **Runtime** -- importing the whole playing stack in a fresh interpreter does
   not pull the Stockfish service or ``chess.engine`` into ``sys.modules``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
ENGINE_DIR = BACKEND_DIR / "app" / "engine"

#: Modules that participate in choosing a move for the network.
PLAYING_MODULES = [
    "encoding.py",
    "network.py",
    "evaluator.py",
    "mcts.py",
    "rules.py",
    "replay.py",
    "selfplay.py",
    "arena.py",
    "watch.py",
]

FORBIDDEN_PREFIXES = ("chess.engine", "app.services", "stockfish")


def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module, node.lineno


@pytest.mark.parametrize("filename", PLAYING_MODULES)
def test_playing_modules_never_import_the_classical_engine(filename):
    path = ENGINE_DIR / filename
    offending = [
        (module, line)
        for module, line in _imported_modules(path)
        if module.startswith(FORBIDDEN_PREFIXES)
    ]
    assert not offending, f"{filename} reaches the classical engine at {offending}"


def test_trainer_only_touches_stockfish_inside_the_benchmark_helper():
    """``train.py`` may measure against Stockfish, but only lazily and locally."""
    path = ENGINE_DIR / "train.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    module_level = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(name.startswith(FORBIDDEN_PREFIXES) for name in module_level)

    benchmark_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            module.startswith("app.services")
            for module, _ in _imported_modules_in(node)
        )
    ]
    assert [f.name for f in benchmark_functions] == ["_run_benchmark"]


def _imported_modules_in(node: ast.AST):
    for child in ast.walk(node):
        if isinstance(child, ast.ImportFrom) and child.module:
            yield child.module, child.lineno
        elif isinstance(child, ast.Import):
            for alias in child.names:
                yield alias.name, child.lineno


def test_playing_stack_does_not_load_stockfish_at_runtime():
    """Importing everything needed to play must not pull in the UCI service.

    ``chess.engine`` itself is deliberately not in the forbidden set here:
    ``chess.pgn`` imports it unconditionally as part of python-chess, so its
    presence in ``sys.modules`` says nothing. What matters is that
    ``app.services.stockfish_service`` -- the only code that can actually spawn
    a Stockfish process -- is never loaded.
    """
    script = (
        "import sys;"
        "import app.engine.mcts, app.engine.selfplay, app.engine.arena;"
        "import app.services.nn_engine;"
        "bad=[m for m in sys.modules"
        " if 'stockfish' in m or m.startswith('app.services.benchmark')];"
        "print(bad)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(BACKEND_DIR),
        capture_output=True,
        text=True,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(BACKEND_DIR)},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]", completed.stdout


def test_neural_move_selection_signature_has_no_engine_hooks():
    """A defensive guard against someone adding a 'use stockfish' flag later."""
    from app.services.nn_engine import NeuralEngineService

    import inspect

    signature = inspect.signature(NeuralEngineService.choose_move)
    names = set(signature.parameters)
    assert not names & {"stockfish", "engine", "assist", "hint"}
