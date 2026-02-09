#!/usr/bin/env python3
"""Clawde — Lichess bot that plays live games via the Lichess Bot API."""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time

import chess
import requests

from clawde.engine import Searcher

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LICHESS_API = "https://lichess.org"
TOKEN_ENV = "LICHESS_BOT_TOKEN"

LOG_FILE = os.environ.get("CLAWDE_LOG_FILE", "clawde.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("clawde")

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _stream_ndjson(url: str, token: str):
    """Yield parsed JSON objects from an NDJSON streaming endpoint."""
    with requests.get(url, headers=_headers(token), stream=True, timeout=30) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Time management
# ---------------------------------------------------------------------------


def calculate_time_limit(
    wtime: int, btime: int, winc: int, binc: int, we_are_white: bool,
    moves_played: int, initial_time: int,
) -> float:
    """Return a time limit in seconds for the engine search.

    Adapts pacing to the time control: thinks quickly in bullet,
    takes its time in classical.
    """
    our_time = wtime if we_are_white else btime
    our_inc = winc if we_are_white else binc

    # Scale pacing to the time control (initial_time is in ms)
    if initial_time <= 60_000:        # ultrabullet (<=1 min)
        target_moves = 50
        min_time = 0.1
    elif initial_time <= 180_000:     # bullet (<=3 min)
        target_moves = 40
        min_time = 0.2
    elif initial_time <= 480_000:     # blitz (<=8 min)
        target_moves = 35
        min_time = 0.5
    elif initial_time <= 900_000:     # rapid (<=15 min)
        target_moves = 30
        min_time = 1.0
    else:                             # classical
        target_moves = 25
        min_time = 2.0

    moves_remaining = max(target_moves - moves_played, 10)
    time_limit = (our_time / (moves_remaining + 10) + our_inc * 0.8) / 1000.0
    time_limit = min(time_limit, our_time / 2000.0)
    time_limit = max(time_limit, min_time)
    return time_limit


# ---------------------------------------------------------------------------
# Game handler (runs in its own thread)
# ---------------------------------------------------------------------------


def handle_game(game_id: str, token: str):
    """Stream a single game and respond with engine moves."""
    log.info("Game %s: starting handler", game_id)
    searcher = Searcher(info_handler=log.debug)
    board = chess.Board()
    we_are_white = True
    initial_time = 300_000  # default 5 min if clock info missing

    try:
        for event in _stream_ndjson(f"{LICHESS_API}/api/bot/game/stream/{game_id}", token):
            etype = event.get("type")

            if etype == "gameFull":
                # Determine our color
                my_id = _get_my_id(token)
                white_id = _nested_id(event, "white")
                we_are_white = (my_id == white_id)
                color_str = "white" if we_are_white else "black"
                log.info("Game %s: playing as %s", game_id, color_str)

                # Extract initial clock for time management pacing
                clock = event.get("clock") or {}
                initial_time = clock.get("initial", 300_000)

                # Reconstruct board from moves so far
                board = chess.Board()
                moves_str = event.get("state", {}).get("moves", "")
                if moves_str:
                    for uci in moves_str.split():
                        board.push_uci(uci)

                state = event.get("state", {})
                _maybe_move(board, searcher, game_id, token, we_are_white, state, initial_time)

            elif etype == "gameState":
                # Rebuild board from full moves string
                board = chess.Board()
                moves_str = event.get("moves", "")
                if moves_str:
                    for uci in moves_str.split():
                        board.push_uci(uci)

                status = event.get("status", "started")
                if status != "started":
                    log.info("Game %s: ended with status %s", game_id, status)
                    return

                _maybe_move(board, searcher, game_id, token, we_are_white, event, initial_time)

            elif etype == "chatLine":
                pass  # Ignore chat

    except requests.exceptions.RequestException as e:
        log.error("Game %s: stream error: %s", game_id, e)
    except Exception as e:
        log.exception("Game %s: unexpected error: %s", game_id, e)

    log.info("Game %s: handler exiting", game_id)


def _nested_id(event: dict, color: str) -> str | None:
    """Extract the player id from a gameFull event."""
    player = event.get(color, {})
    return player.get("id")


def _maybe_move(
    board: chess.Board,
    searcher: Searcher,
    game_id: str,
    token: str,
    we_are_white: bool,
    state: dict,
    initial_time: int,
):
    """If it's our turn and the game isn't over, search and play a move."""
    if board.is_game_over():
        return

    our_turn = (board.turn == chess.WHITE) == we_are_white
    if not our_turn:
        return

    moves_played = board.fullmove_number
    wtime = state.get("wtime", 60000)
    btime = state.get("btime", 60000)
    winc = state.get("winc", 0)
    binc = state.get("binc", 0)

    time_limit = calculate_time_limit(wtime, btime, winc, binc, we_are_white, moves_played, initial_time)
    log.info("Game %s: thinking (%.1fs limit, %dms on clock)",
             game_id, time_limit,
             wtime if we_are_white else btime)

    move = searcher.search(board, time_limit=time_limit)
    uci = move.uci()
    log.info("Game %s: playing %s", game_id, uci)

    try:
        resp = requests.post(
            f"{LICHESS_API}/api/bot/game/{game_id}/move/{uci}",
            headers=_headers(token),
            timeout=10,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        log.error("Game %s: failed to send move %s: %s", game_id, uci, e)


# ---------------------------------------------------------------------------
# Account helpers
# ---------------------------------------------------------------------------

_my_id_cache: str | None = None


def _get_my_id(token: str) -> str:
    """Fetch and cache our Lichess user ID."""
    global _my_id_cache
    if _my_id_cache is None:
        resp = requests.get(f"{LICHESS_API}/api/account", headers=_headers(token), timeout=10)
        resp.raise_for_status()
        _my_id_cache = resp.json()["id"]
    return _my_id_cache


# ---------------------------------------------------------------------------
# Event loop (main thread)
# ---------------------------------------------------------------------------


def event_loop(token: str):
    """Listen for incoming challenges and game starts."""
    while True:
        try:
            log.info("Listening for events...")
            for event in _stream_ndjson(f"{LICHESS_API}/api/stream/event", token):
                etype = event.get("type")

                if etype == "challenge":
                    challenge = event["challenge"]
                    cid = challenge["id"]
                    challenger = challenge.get("challenger", {}).get("name", "?")
                    variant = challenge.get("variant", {}).get("key", "?")

                    if variant != "standard":
                        log.info("Declining challenge %s from %s (variant: %s)", cid, challenger, variant)
                        try:
                            requests.post(
                                f"{LICHESS_API}/api/challenge/{cid}/decline",
                                headers=_headers(token),
                                json={"reason": "variant"},
                                timeout=10,
                            )
                        except requests.exceptions.RequestException:
                            pass
                        continue

                    log.info("Accepting challenge %s from %s", cid, challenger)
                    try:
                        resp = requests.post(
                            f"{LICHESS_API}/api/challenge/{cid}/accept",
                            headers=_headers(token),
                            timeout=10,
                        )
                        resp.raise_for_status()
                    except requests.exceptions.RequestException as e:
                        log.error("Failed to accept challenge %s: %s", cid, e)

                elif etype == "gameStart":
                    game_id = event["game"]["gameId"]
                    log.info("Game started: %s", game_id)
                    t = threading.Thread(
                        target=handle_game,
                        args=(game_id, token),
                        daemon=True,
                        name=f"game-{game_id}",
                    )
                    t.start()

                elif etype == "gameFinish":
                    game_id = event["game"]["gameId"]
                    log.info("Game finished: %s", game_id)

        except requests.exceptions.RequestException as e:
            log.error("Event stream disconnected: %s — reconnecting in 5s", e)
            time.sleep(5)
        except Exception as e:
            log.exception("Event loop error: %s — reconnecting in 5s", e)
            time.sleep(5)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    token = os.environ.get(TOKEN_ENV)
    if not token:
        print(f"Error: set the {TOKEN_ENV} environment variable.")
        print(f"  export {TOKEN_ENV}='lip_...'")
        sys.exit(1)

    # Verify the token works
    try:
        username = _get_my_id(token)
        log.info("Authenticated as %s", username)
    except requests.exceptions.RequestException as e:
        print(f"Error: failed to authenticate with Lichess: {e}")
        sys.exit(1)

    event_loop(token)
