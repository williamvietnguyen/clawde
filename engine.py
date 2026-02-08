#!/usr/bin/env python3
"""Clawde — a UCI-compatible chess engine written in Python."""

import time
import threading
from dataclasses import dataclass
from enum import IntEnum

import chess
import chess.polyglot

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MATE_SCORE = 100_000
INF = MATE_SCORE + 1
MAX_DEPTH = 64
TT_SIZE = 1 << 20  # ~1M entries

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Piece-square tables (from White's perspective, a1=index 0).
# Flipped for Black at runtime.

PST_PAWN = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

PST_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

PST_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

PST_ROOK = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

PST_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

PST_KING_MG = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

PST_KING_EG = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]

# Tables indexed by piece type; king uses MG table (EG blended at eval time).
PST = {
    chess.PAWN: PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK: PST_ROOK,
    chess.QUEEN: PST_QUEEN,
    chess.KING: PST_KING_MG,
}


def _flip_sq(sq: int) -> int:
    """Mirror a square vertically (for Black's PST lookup)."""
    return sq ^ 56


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _game_phase(board: chess.Board) -> float:
    """Return 0.0 (endgame) … 1.0 (opening) based on remaining material."""
    phase = 0
    phase += chess.popcount(board.knights) * 1
    phase += chess.popcount(board.bishops) * 1
    phase += chess.popcount(board.rooks) * 2
    phase += chess.popcount(board.queens) * 4
    return min(phase / 24.0, 1.0)


def evaluate(board: chess.Board) -> int:
    """Static evaluation from the side-to-move's point of view."""
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0

    phase = _game_phase(board)
    score = 0

    for color in chess.COLORS:
        sign = 1 if color == chess.WHITE else -1
        for pt in chess.PIECE_TYPES:
            for sq in board.pieces(pt, color):
                # Material
                score += sign * PIECE_VALUES[pt]
                # PST
                idx = sq if color == chess.WHITE else _flip_sq(sq)
                score += sign * PST[pt][idx]
                # King endgame PST interpolation
                if pt == chess.KING:
                    eg_bonus = PST_KING_EG[idx] - PST_KING_MG[idx]
                    score += sign * int(eg_bonus * (1.0 - phase))

        # Bishop pair bonus
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            score += sign * 30

    # Tempo bonus
    score += 10

    return score if board.turn == chess.WHITE else -score


# ---------------------------------------------------------------------------
# Transposition table
# ---------------------------------------------------------------------------

class TTFlag(IntEnum):
    EXACT = 0
    ALPHA = 1  # upper bound
    BETA = 2   # lower bound


@dataclass(slots=True)
class TTEntry:
    key: int = 0
    depth: int = 0
    score: int = 0
    flag: int = TTFlag.EXACT
    best_move: chess.Move | None = None


class TranspositionTable:
    def __init__(self, size: int = TT_SIZE):
        self.size = size
        self.table: list[TTEntry | None] = [None] * size

    def probe(self, key: int) -> TTEntry | None:
        entry = self.table[key % self.size]
        if entry is not None and entry.key == key:
            return entry
        return None

    def store(self, key: int, depth: int, score: int, flag: int,
              best_move: chess.Move | None):
        idx = key % self.size
        entry = self.table[idx]
        # Always-replace with depth-preferred
        if entry is None or entry.key != key or depth >= entry.depth:
            self.table[idx] = TTEntry(key, depth, score, flag, best_move)

    def clear(self):
        self.table = [None] * self.size


# ---------------------------------------------------------------------------
# Move ordering
# ---------------------------------------------------------------------------

MVV_LVA_SCORES = {}
for victim in chess.PIECE_TYPES:
    for attacker in chess.PIECE_TYPES:
        MVV_LVA_SCORES[(victim, attacker)] = (
            PIECE_VALUES[victim] * 10 - PIECE_VALUES.get(attacker, 0)
        )


class MoveOrderer:
    def __init__(self):
        self.killer_moves: list[list[chess.Move | None]] = [
            [None, None] for _ in range(MAX_DEPTH)
        ]
        self.history: list[list[int]] = [
            [0] * 64 for _ in range(64)
        ]

    def clear(self):
        self.killer_moves = [[None, None] for _ in range(MAX_DEPTH)]
        self.history = [[0] * 64 for _ in range(64)]

    def store_killer(self, move: chess.Move, ply: int):
        if move != self.killer_moves[ply][0]:
            self.killer_moves[ply][1] = self.killer_moves[ply][0]
            self.killer_moves[ply][0] = move

    def update_history(self, move: chess.Move, depth: int):
        self.history[move.from_square][move.to_square] += depth * depth

    def score_move(self, move: chess.Move, board: chess.Board, ply: int,
                   tt_move: chess.Move | None) -> int:
        # TT move first
        if move == tt_move:
            return 1_000_000

        # Captures scored by MVV-LVA
        if board.is_capture(move):
            victim_sq = move.to_square
            victim = board.piece_type_at(victim_sq)
            # En-passant
            if victim is None:
                victim = chess.PAWN
            attacker = board.piece_type_at(move.from_square)
            return 500_000 + MVV_LVA_SCORES.get((victim, attacker), 0)

        # Promotions
        if move.promotion:
            return 400_000 + PIECE_VALUES.get(move.promotion, 0)

        # Killers
        if ply < MAX_DEPTH:
            if move == self.killer_moves[ply][0]:
                return 300_000
            if move == self.killer_moves[ply][1]:
                return 299_000

        # History heuristic
        return self.history[move.from_square][move.to_square]

    def ordered_moves(self, board: chess.Board, ply: int,
                      tt_move: chess.Move | None) -> list[chess.Move]:
        moves = list(board.legal_moves)
        moves.sort(key=lambda m: self.score_move(m, board, ply, tt_move),
                   reverse=True)
        return moves


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class Searcher:
    def __init__(self):
        self.tt = TranspositionTable()
        self.orderer = MoveOrderer()
        self.nodes = 0
        self.stop_event = threading.Event()
        self.best_move: chess.Move | None = None
        self.start_time = 0.0
        self.time_limit: float | None = None

    def _check_time(self):
        if self.time_limit is not None and self.nodes & 2047 == 0:
            if time.time() - self.start_time >= self.time_limit:
                self.stop_event.set()

    def quiesce(self, board: chess.Board, alpha: int, beta: int) -> int:
        """Quiescence search — resolve captures to avoid horizon effect."""
        self.nodes += 1
        if self.stop_event.is_set():
            return 0

        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Delta pruning threshold
        BIG_DELTA = PIECE_VALUES[chess.QUEEN]

        for move in board.generate_legal_captures():
            # Delta pruning
            captured = board.piece_type_at(move.to_square)
            if captured is None:
                captured = chess.PAWN  # en passant
            if stand_pat + PIECE_VALUES[captured] + 200 < alpha and not move.promotion:
                continue

            if stand_pat + BIG_DELTA < alpha:
                break

            board.push(move)
            score = -self.quiesce(board, -beta, -alpha)
            board.pop()

            if self.stop_event.is_set():
                return 0

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def _alpha_beta(self, board: chess.Board, depth: int, alpha: int,
                    beta: int, ply: int) -> int:
        self._check_time()
        if self.stop_event.is_set():
            return 0

        self.nodes += 1
        is_root = ply == 0
        is_pv = beta - alpha > 1

        # Draw detection
        if not is_root and (board.is_repetition(2) or board.halfmove_clock >= 100):
            return 0

        # Probe TT
        key = chess.polyglot.zobrist_hash(board)
        tt_entry = self.tt.probe(key)
        tt_move = None
        if tt_entry is not None:
            tt_move = tt_entry.best_move
            if not is_pv and tt_entry.depth >= depth:
                score = tt_entry.score
                if tt_entry.flag == TTFlag.EXACT:
                    return score
                if tt_entry.flag == TTFlag.BETA and score >= beta:
                    return score
                if tt_entry.flag == TTFlag.ALPHA and score <= alpha:
                    return score

        # Quiescence at leaf
        if depth <= 0:
            return self.quiesce(board, alpha, beta)

        in_check = board.is_check()

        # Null-move pruning (skip when in check, at root, or in zugzwang-prone endings)
        if (not is_pv and not in_check and depth >= 3 and not is_root
                and _game_phase(board) > 0.2):
            R = 2 + (depth >= 6)
            board.push(chess.Move.null())
            score = -self._alpha_beta(board, depth - 1 - R, -beta, -beta + 1, ply + 1)
            board.pop()
            if self.stop_event.is_set():
                return 0
            if score >= beta:
                return beta

        moves = self.orderer.ordered_moves(board, ply, tt_move)

        if not moves:
            if in_check:
                return -MATE_SCORE + ply
            return 0  # stalemate

        best_score = -INF
        best_move = moves[0]
        flag = TTFlag.ALPHA

        for i, move in enumerate(moves):
            # Check extension
            ext = 1 if board.gives_check(move) else 0

            board.push(move)

            # Late Move Reductions
            if (i >= 4 and depth >= 3 and not in_check
                    and not board.is_check() and not board.is_capture(move)
                    and not move.promotion):
                # Reduced depth search
                score = -self._alpha_beta(board, depth - 2 + ext, -alpha - 1, -alpha, ply + 1)
                if score <= alpha:
                    board.pop()
                    continue

            # PVS
            if i == 0:
                score = -self._alpha_beta(board, depth - 1 + ext, -beta, -alpha, ply + 1)
            else:
                score = -self._alpha_beta(board, depth - 1 + ext, -alpha - 1, -alpha, ply + 1)
                if alpha < score < beta:
                    score = -self._alpha_beta(board, depth - 1 + ext, -beta, -alpha, ply + 1)

            board.pop()

            if self.stop_event.is_set():
                return 0

            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:
                    alpha = score
                    flag = TTFlag.EXACT
                    if score >= beta:
                        flag = TTFlag.BETA
                        # Update killer / history for quiet moves
                        if not board.is_capture(move):
                            self.orderer.store_killer(move, ply)
                            self.orderer.update_history(move, depth)
                        self.tt.store(key, depth, score, flag, best_move)
                        return score

        self.tt.store(key, depth, best_score, flag, best_move)

        if is_root:
            self.best_move = best_move

        return best_score

    def search(self, board: chess.Board, max_depth: int = MAX_DEPTH,
               time_limit: float | None = None) -> chess.Move:
        """Iterative-deepening search. Returns the best move found."""
        self.nodes = 0
        self.stop_event.clear()
        self.start_time = time.time()
        self.time_limit = time_limit
        self.best_move = None

        best_for_iter = None

        for depth in range(1, max_depth + 1):
            score = self._alpha_beta(board, depth, -INF, INF, 0)

            if self.stop_event.is_set():
                break

            best_for_iter = self.best_move
            elapsed = time.time() - self.start_time
            nps = int(self.nodes / elapsed) if elapsed > 0 else 0

            # Convert score to UCI format
            if abs(score) >= MATE_SCORE - MAX_DEPTH:
                mate_in = (MATE_SCORE - abs(score) + 1) // 2
                if score < 0:
                    mate_in = -mate_in
                score_str = f"score mate {mate_in}"
            else:
                score_str = f"score cp {score}"

            print(f"info depth {depth} {score_str} nodes {self.nodes} "
                  f"nps {nps} time {int(elapsed * 1000)} "
                  f"pv {best_for_iter.uci() if best_for_iter else '(none)'}",
                  flush=True)

            # If we found a forced mate, stop searching deeper
            if abs(score) >= MATE_SCORE - MAX_DEPTH:
                break

        return best_for_iter or list(board.legal_moves)[0]


# ---------------------------------------------------------------------------
# UCI protocol
# ---------------------------------------------------------------------------

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


if __name__ == "__main__":
    UCI().run()
