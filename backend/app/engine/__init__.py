"""Neural chess engine.

IMPORTANT ARCHITECTURAL RULE
---------------------------
Nothing in this package may import from ``app.services.stockfish_service`` (or
otherwise reach a classical engine). The neural network must learn purely from
self-play; the Stockfish-backed analysis is a *human-only* affordance exposed by
the API layer. ``tests/test_engine_isolation.py`` enforces this mechanically.
"""
