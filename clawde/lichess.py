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

from clawde import book
from clawde.engine import Searcher

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LICHESS_API = "https://lichess.org"
TOKEN_ENV = "LICHESS_BOT_TOKEN"

# Accepted Lichess speed categories.  Challenges outside this set are declined.
# Valid values: ultraBullet, bullet, blitz, rapid, classical, correspondence
ACCEPTED_SPEEDS: set[str] = {
    "bullet", "blitz", "rapid", "classical",
}

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
    our_time_ms: int,
    our_inc_ms: int,
    initial_time_ms: int,
    moves_played: int,
    moves_out_of_book: int,
) -> float:
    """Return a time limit in seconds for the engine search.

    Uses a simple model: spread remaining time evenly over estimated
    remaining moves, plus most of the increment.  After leaving the
    opening book the engine ramps up gradually over several moves so
    the transition from instant book moves to real search feels natural.
    """
    # Estimate how many moves remain based on time control.
    # Faster games → more moves (less time to find wins → longer games).
    if initial_time_ms <= 60_000:          # ultrabullet
        expected_total = 100
        floor_s = 0.05
    elif initial_time_ms <= 180_000:       # bullet
        expected_total = 80
        floor_s = 0.1
    elif initial_time_ms <= 480_000:       # blitz
        expected_total = 60
        floor_s = 0.25
    elif initial_time_ms <= 900_000:       # rapid
        expected_total = 50
        floor_s = 0.5
    else:                                  # classical
        expected_total = 40
        floor_s = 1.0

    moves_left = max(expected_total - moves_played, 10)

    # Base allocation: even share of remaining time + most of increment.
    # The 0.9 factor keeps a 10% buffer so we don't flag.
    base_s = (our_time_ms * 0.9 / moves_left + our_inc_ms * 0.8) / 1000.0

    # Ramp up after leaving book: 40% → 100% over 5 moves
    if moves_out_of_book < 5:
        ramp = 0.4 + 0.12 * moves_out_of_book
        base_s *= ramp

    # Per-move cap — a Python engine doesn't gain much depth from
    # extra time, so keep it moving at a natural pace for the format.
    if initial_time_ms <= 60_000:            # ultrabullet
        cap_s = 0.15
    elif initial_time_ms <= 180_000:         # bullet
        cap_s = 0.8
    elif initial_time_ms <= 480_000:         # blitz
        cap_s = 2.5
    elif initial_time_ms <= 900_000:         # rapid
        cap_s = 8.0
    else:                                    # classical
        cap_s = 20.0

    # Also never burn more than 15% of remaining time on one move
    ceiling_s = min(cap_s, our_time_ms * 0.15 / 1000.0)
    base_s = min(base_s, ceiling_s)

    # Floor: always think at least a minimum amount
    return max(base_s, floor_s)


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
    book_exit_ply: int | None = None  # ply when we first left the book

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
                book_exit_ply = _maybe_move(
                    board, searcher, game_id, token, we_are_white,
                    state, initial_time, book_exit_ply,
                )

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

                book_exit_ply = _maybe_move(
                    board, searcher, game_id, token, we_are_white,
                    event, initial_time, book_exit_ply,
                )

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
    book_exit_ply: int | None,
) -> int | None:
    """If it's our turn and the game isn't over, play a move.

    Returns the updated *book_exit_ply* so the caller can track it.
    """
    if board.is_game_over():
        return book_exit_ply

    our_turn = (board.turn == chess.WHITE) == we_are_white
    if not our_turn:
        return book_exit_ply

    current_ply = len(board.move_stack)

    # --- Try the opening book first ---
    book_move = book.probe(board)
    if book_move is not None and book_move in board.legal_moves:
        uci = book_move.uci()
        log.info("Game %s: book move %s", game_id, uci)
        _send_move(game_id, token, uci)
        return book_exit_ply  # still in book

    # First time out of book — record the ply
    if book_exit_ply is None:
        book_exit_ply = current_ply
        log.info("Game %s: leaving book at ply %d", game_id, current_ply)

    # --- Time management ---
    moves_played = board.fullmove_number
    our_time = state.get("wtime", 60000) if we_are_white else state.get("btime", 60000)
    our_inc = state.get("winc", 0) if we_are_white else state.get("binc", 0)
    moves_out_of_book = current_ply - book_exit_ply

    time_limit = calculate_time_limit(
        our_time, our_inc, initial_time, moves_played, moves_out_of_book,
    )
    log.info("Game %s: thinking (%.2fs limit, %dms on clock, %d moves out of book)",
             game_id, time_limit, our_time, moves_out_of_book)

    move = searcher.search(board, time_limit=time_limit)
    uci = move.uci()
    log.info("Game %s: playing %s", game_id, uci)
    _send_move(game_id, token, uci)
    return book_exit_ply


def _send_move(game_id: str, token: str, uci: str):
    """POST a move to the Lichess API."""
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

                    speed = challenge.get("speed", "?")

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

                    if speed not in ACCEPTED_SPEEDS:
                        log.info("Declining challenge %s from %s (speed: %s)", cid, challenger, speed)
                        try:
                            requests.post(
                                f"{LICHESS_API}/api/challenge/{cid}/decline",
                                headers=_headers(token),
                                json={"reason": "timeControl"},
                                timeout=10,
                            )
                        except requests.exceptions.RequestException:
                            pass
                        continue

                    log.info("Accepting challenge %s from %s (%s)", cid, challenger, speed)
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
