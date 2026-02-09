"""Built-in opening book for Clawde.

Covers major openings (~20 lines, 6–10 moves deep) for both colors.
Positions are indexed by Polyglot Zobrist hash; when multiple candidate
moves exist the engine picks one at random weighted by popularity.
"""
from __future__ import annotations

import random

import chess
import chess.polyglot

# ---------------------------------------------------------------------------
# Opening repertoire
# ---------------------------------------------------------------------------
# Each entry: (space-separated UCI moves, weight).
# Higher weight → more likely to be chosen when alternatives exist.

_OPENING_LINES: list[tuple[str, int]] = [
    # ===== 1.e4 =====

    # Ruy Lopez, Closed (Morphy Defense)
    ("e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3 c6b8 d2d4", 10),

    # Ruy Lopez, Berlin
    ("e2e4 e7e5 g1f3 b8c6 f1b5 g8f6 e1g1 f6e4 d2d4 e4d6 b5c6 d7c6 d4e5 d6f5 d1d8 e8d8", 8),

    # Italian Game (Quiet)
    ("e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d3 f8c5 e1g1 d7d6 c2c3 e8g8", 8),

    # Scotch Game
    ("e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6 d4c6 b7c6 e4e5 d8e7 d1e2 f6d5", 6),

    # Petroff Defense
    ("e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4 d2d4 d6d5 f1d3 b8c6 e1g1 f8e7 c2c4", 6),

    # Sicilian Najdorf
    ("e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 f1e2 e7e5 d4b3 f8e7 e1g1 e8g8", 10),

    # Sicilian Dragon
    ("e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 g7g6 c1e3 f8g7 f2f3 e8g8 d1d2 b8c6", 8),

    # Sicilian Scheveningen
    ("e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e6 f1e2 f8e7 e1g1 e8g8 f2f4 a7a6", 7),

    # French Tarrasch
    ("e2e4 e7e6 d2d4 d7d5 b1d2 g8f6 e4e5 f6d7 f1d3 c7c5 c2c3 b8c6 g1e2 c5d4 c3d4", 7),

    # Caro-Kann Classical
    ("e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5 e4g3 f5g6 h2h4 h7h6 g1f3 b8d7", 7),

    # Caro-Kann Advance
    ("e2e4 c7c6 d2d4 d7d5 e4e5 c8f5 g1f3 e7e6 f1e2 c6c5 c1e3 b8d7", 6),

    # ===== 1.d4 =====

    # Queen's Gambit Declined
    ("d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 g1f3 b8d7 a1c1 c7c6", 9),

    # Queen's Gambit Accepted
    ("d2d4 d7d5 c2c4 d5c4 g1f3 g8f6 e2e3 e7e6 f1c4 c7c5 e1g1 a7a6", 7),

    # Nimzo-Indian
    ("d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 d1c2 e8g8 a2a3 b4c3 c2c3 b7b6 c1g5", 8),

    # Queen's Indian
    ("d2d4 g8f6 c2c4 e7e6 g1f3 b7b6 g2g3 c8b7 f1g2 f8e7 e1g1 e8g8 b1c3 f6e4", 7),

    # King's Indian Classical
    ("d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2 e7e5 e1g1 b8c6 d4d5 c6e7", 9),

    # King's Indian Saemisch
    ("d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 f2f3 e8g8 c1e3 e7e5 d4d5 c7c5", 7),

    # Gruenfeld Exchange
    ("d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5 e2e4 d5c3 b2c3 f8g7 g1f3 c7c5 f1e2 e8g8", 8),

    # Slav Defense
    ("d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4 a2a4 c8f5 e2e3 e7e6 f1c4", 7),

    # Catalan
    ("d2d4 g8f6 c2c4 e7e6 g2g3 d7d5 f1g2 f8e7 g1f3 e8g8 e1g1 d5c4 d1c2 a7a6 a2a4", 8),

    # ===== 1.c4 / 1.Nf3 =====

    # English Opening
    ("c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3 d7d5 c4d5 f6d5 f1g2 d5b6", 6),

    # London System
    ("d2d4 d7d5 c1f4 g8f6 e2e3 c7c5 c2c3 b8c6 b1d2 e7e6 g1f3 f8d6", 6),
]


# ---------------------------------------------------------------------------
# Build the lookup table at import time
# ---------------------------------------------------------------------------

# hash -> [(move, cumulative_weight), ...]
_BOOK: dict[int, list[tuple[chess.Move, int]]] = {}


def _build() -> None:
    for line_uci, weight in _OPENING_LINES:
        board = chess.Board()
        for token in line_uci.split():
            key = chess.polyglot.zobrist_hash(board)
            move = chess.Move.from_uci(token)
            entries = _BOOK.setdefault(key, [])
            # Merge weight if this move already recorded for this position
            for i, (m, w) in enumerate(entries):
                if m == move:
                    entries[i] = (m, w + weight)
                    break
            else:
                entries.append((move, weight))
            board.push(move)


_build()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def probe(board: chess.Board) -> chess.Move | None:
    """Return a book move for *board*, or ``None`` if out of book."""
    key = chess.polyglot.zobrist_hash(board)
    entries = _BOOK.get(key)
    if not entries:
        return None
    moves, weights = zip(*entries)
    return random.choices(moves, weights=weights, k=1)[0]
