"""UCI protocol interface for Clawde."""
from __future__ import annotations

import threading

import chess

from clawde import book
from clawde.engine import MAX_DEPTH, Searcher


class UCI:
    def __init__(self):
        self.board = chess.Board()
        self.searcher = Searcher()
        self.search_thread: threading.Thread | None = None

    def _send(self, msg: str):
        print(msg, flush=True)

    def _handle_position(self, tokens: list[str]):
        idx = 0
        if tokens[idx] == "startpos":
            self.board = chess.Board()
            idx += 1
        elif tokens[idx] == "fen":
            idx += 1
            fen_parts = []
            while idx < len(tokens) and tokens[idx] != "moves":
                fen_parts.append(tokens[idx])
                idx += 1
            self.board = chess.Board(" ".join(fen_parts))

        if idx < len(tokens) and tokens[idx] == "moves":
            idx += 1
            while idx < len(tokens):
                self.board.push_uci(tokens[idx])
                idx += 1

    def _handle_go(self, tokens: list[str]):
        max_depth = MAX_DEPTH
        time_limit = None
        wtime = btime = None
        winc = binc = 0
        movestogo = 30
        movetime = None

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "depth" and i + 1 < len(tokens):
                max_depth = int(tokens[i + 1]); i += 2
            elif tok == "wtime" and i + 1 < len(tokens):
                wtime = int(tokens[i + 1]); i += 2
            elif tok == "btime" and i + 1 < len(tokens):
                btime = int(tokens[i + 1]); i += 2
            elif tok == "winc" and i + 1 < len(tokens):
                winc = int(tokens[i + 1]); i += 2
            elif tok == "binc" and i + 1 < len(tokens):
                binc = int(tokens[i + 1]); i += 2
            elif tok == "movestogo" and i + 1 < len(tokens):
                movestogo = int(tokens[i + 1]); i += 2
            elif tok == "movetime" and i + 1 < len(tokens):
                movetime = int(tokens[i + 1]); i += 2
            elif tok == "infinite":
                max_depth = MAX_DEPTH; i += 1
            else:
                i += 1

        # Time management
        if movetime is not None:
            time_limit = movetime / 1000.0
        elif wtime is not None and btime is not None:
            our_time = wtime if self.board.turn == chess.WHITE else btime
            our_inc = winc if self.board.turn == chess.WHITE else binc
            # Use a fraction of remaining time + most of increment
            time_limit = (our_time / (movestogo + 5) + our_inc * 0.8) / 1000.0
            # Never use more than half our remaining time
            time_limit = min(time_limit, our_time / 2000.0)
            # At least 50ms
            time_limit = max(time_limit, 0.05)

        def search_and_report():
            book_move = book.probe(self.board)
            if book_move is not None and book_move in self.board.legal_moves:
                self._send(f"info string book move {book_move.uci()}")
                self._send(f"bestmove {book_move.uci()}")
                return
            move = self.searcher.search(self.board, max_depth, time_limit)
            self._send(f"bestmove {move.uci()}")

        self.search_thread = threading.Thread(target=search_and_report, daemon=True)
        self.search_thread.start()

    def run(self):
        while True:
            try:
                line = input()
            except EOFError:
                break

            line = line.strip()
            if not line:
                continue

            tokens = line.split()
            cmd = tokens[0]

            if cmd == "uci":
                self._send("id name Clawde 1.0")
                self._send("id author Claude")
                self._send("uciok")

            elif cmd == "isready":
                self._send("readyok")

            elif cmd == "ucinewgame":
                self.searcher.tt.clear()
                self.searcher.orderer.clear()

            elif cmd == "position":
                self._handle_position(tokens[1:])

            elif cmd == "go":
                self._handle_go(tokens[1:])

            elif cmd == "stop":
                self.searcher.stop_event.set()
                if self.search_thread:
                    self.search_thread.join()

            elif cmd == "quit":
                self.searcher.stop_event.set()
                if self.search_thread:
                    self.search_thread.join()
                break
