#!/usr/bin/env python3
"""Clawde — a UCI chess engine and Lichess bot.

Usage:
    python main.py uci       Start the UCI interface (default)
    python main.py lichess   Connect to Lichess and play live games
"""
from __future__ import annotations

import sys


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "uci"

    if mode == "uci":
        from clawde.uci import UCI
        UCI().run()

    elif mode == "lichess":
        from clawde.lichess import main as lichess_main
        lichess_main()

    else:
        print(f"Unknown mode: {mode!r}")
        print("Usage: python main.py [uci | lichess]")
        sys.exit(1)


if __name__ == "__main__":
    main()
