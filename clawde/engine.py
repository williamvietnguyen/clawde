#!/usr/bin/env python3
"""Clawde — a UCI-compatible chess engine written in Python."""
from __future__ import annotations

import math
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
# Pawn structure tables (precomputed bitmasks)
# ---------------------------------------------------------------------------

# Passed pawn bonus by relative rank (0=back rank, 7=promotion rank).
# Heavily weighted toward endgame — a passed pawn on the 6th is almost a piece.
PASSED_PAWN_BONUS_MG = [0, 5, 10, 15, 25, 40, 60, 0]
PASSED_PAWN_BONUS_EG = [0, 10, 20, 35, 60, 100, 160, 0]

ISOLATED_PAWN_PENALTY = 15
DOUBLED_PAWN_PENALTY = 10

# Mobility: centipawns per square above/below the baseline.
_MOBILITY_WEIGHT   = {chess.KNIGHT: 4, chess.BISHOP: 5, chess.ROOK: 3, chess.QUEEN: 1}
_MOBILITY_BASELINE = {chess.KNIGHT: 4, chess.BISHOP: 6, chess.ROOK: 7, chess.QUEEN: 14}

# File bitboards
_FILE_BB = [0] * 8
for _f in range(8):
    for _r in range(8):
        _FILE_BB[_f] |= 1 << chess.square(_f, _r)

# Adjacent file bitboards (for isolated pawn detection)
_ADJ_FILE_BB = [0] * 8
for _f in range(8):
    if _f > 0:
        _ADJ_FILE_BB[_f] |= _FILE_BB[_f - 1]
    if _f < 7:
        _ADJ_FILE_BB[_f] |= _FILE_BB[_f + 1]

# Front span: squares on same + adjacent files ahead of a pawn.
# If no enemy pawn is in this mask, the pawn is passed.
# Index: [color][square].  WHITE=True=1, BLACK=False=0.
_FRONT_SPAN = [[0] * 64, [0] * 64]
for _sq in range(64):
    _f = chess.square_file(_sq)
    _r = chess.square_rank(_sq)
    for _file in range(max(0, _f - 1), min(7, _f + 1) + 1):
        for _rank in range(_r + 1, 8):
            _FRONT_SPAN[1][_sq] |= 1 << chess.square(_file, _rank)
        for _rank in range(0, _r):
            _FRONT_SPAN[0][_sq] |= 1 << chess.square(_file, _rank)

# File ahead: squares on the SAME file ahead of a pawn (for doubled pawn detection).
# Only the rear pawn in a doubled pair gets penalized.
_FILE_AHEAD = [[0] * 64, [0] * 64]
for _sq in range(64):
    _f = chess.square_file(_sq)
    _r = chess.square_rank(_sq)
    for _rank in range(_r + 1, 8):
        _FILE_AHEAD[1][_sq] |= 1 << chess.square(_f, _rank)
    for _rank in range(0, _r):
        _FILE_AHEAD[0][_sq] |= 1 << chess.square(_f, _rank)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _king_safety(board: chess.Board, color: chess.Color, phase: float) -> int:
    """Return a king safety penalty for *color* (negative = king in danger).

    Evaluates pawn shield, open files near the king, and the number / weight
    of enemy pieces attacking the king zone.  Scaled by game phase so it
    fades out in endgames where the king should be active instead.
    """
    # King safety is irrelevant once most pieces are off the board.
    if phase < 0.15:
        return 0

    king_sq = board.king(color)
    if king_sq is None:
        return 0

    penalty = 0
    enemy = not color
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    our_pawns_bb = int(board.pieces(chess.PAWN, color))
    their_pawns_bb = int(board.pieces(chess.PAWN, enemy))

    # --- Pawn shield ---
    # Check 3 squares directly in front of the king (same file + neighbours).
    pawn_dir = 1 if color == chess.WHITE else -1
    for df in (-1, 0, 1):
        f = king_file + df
        if not (0 <= f <= 7):
            continue
        r1 = king_rank + pawn_dir
        if 0 <= r1 <= 7 and (1 << chess.square(f, r1)) & our_pawns_bb:
            continue  # shield pawn in place
        r2 = king_rank + 2 * pawn_dir
        if 0 <= r2 <= 7 and (1 << chess.square(f, r2)) & our_pawns_bb:
            penalty += 10  # pawn advanced one square — mild weakness
            continue
        penalty += 25  # shield pawn missing entirely

    # --- Open / semi-open files near king ---
    for f in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
        file_bb = _FILE_BB[f]
        if not ((our_pawns_bb | their_pawns_bb) & file_bb):
            penalty += 25  # fully open file
        elif not (our_pawns_bb & file_bb):
            penalty += 15  # semi-open (no friendly pawn)

    # --- Enemy attackers on king zone ---
    king_zone_bb = board.attacks_mask(king_sq) | (1 << king_sq)

    attack_weight = 0
    num_attackers = 0
    for pt, weight in ((chess.KNIGHT, 2), (chess.BISHOP, 2),
                        (chess.ROOK, 3), (chess.QUEEN, 5)):
        for sq in board.pieces(pt, enemy):
            if board.attacks_mask(sq) & king_zone_bb:
                num_attackers += 1
                attack_weight += weight

    # Two or more attackers are disproportionately dangerous.
    if num_attackers >= 2:
        penalty += attack_weight * num_attackers

    return -int(penalty * phase)


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

        # Mobility: reward pieces with more available squares
        friendly_bb = int(board.occupied_co[color])
        for pt, weight in _MOBILITY_WEIGHT.items():
            baseline = _MOBILITY_BASELINE[pt]
            for sq in board.pieces(pt, color):
                mob = chess.popcount(board.attacks_mask(sq) & ~friendly_bb)
                score += sign * weight * (mob - baseline)

        # Pawn structure
        our_pawns = board.pieces(chess.PAWN, color)
        our_pawns_bb = int(our_pawns)
        their_pawns_bb = int(board.pieces(chess.PAWN, not color))

        for sq in our_pawns:
            file = chess.square_file(sq)
            rel_rank = chess.square_rank(sq) if color == chess.WHITE else 7 - chess.square_rank(sq)

            # Passed pawn (no enemy pawns ahead on same or adjacent files)
            if not (their_pawns_bb & _FRONT_SPAN[color][sq]):
                bonus = int(PASSED_PAWN_BONUS_MG[rel_rank] * phase
                            + PASSED_PAWN_BONUS_EG[rel_rank] * (1.0 - phase))
                score += sign * bonus

            # Isolated pawn (no friendly pawns on adjacent files)
            if not (our_pawns_bb & _ADJ_FILE_BB[file]):
                score -= sign * ISOLATED_PAWN_PENALTY

            # Doubled pawn (friendly pawn ahead on same file)
            if our_pawns_bb & _FILE_AHEAD[color][sq]:
                score -= sign * DOUBLED_PAWN_PENALTY

        # King safety
        score += sign * _king_safety(board, color, phase)

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


@dataclass
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
# Static Exchange Evaluation (SEE)
# ---------------------------------------------------------------------------

def _sliding_attacks(sq: int, occ: int, rook: bool, bishop: bool) -> int:
    """Compute sliding-piece attack bitboard from *sq* given *occ*."""
    attacks = 0
    f0, r0 = chess.square_file(sq), chess.square_rank(sq)
    dirs: tuple[tuple[int, int], ...]
    if rook and bishop:
        dirs = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))
    elif rook:
        dirs = ((1,0),(-1,0),(0,1),(0,-1))
    else:
        dirs = ((1,1),(1,-1),(-1,1),(-1,-1))
    for df, dr in dirs:
        f, r = f0 + df, r0 + dr
        while 0 <= f <= 7 and 0 <= r <= 7:
            bit = 1 << chess.square(f, r)
            attacks |= bit
            if occ & bit:
                break
            f += df
            r += dr
    return attacks


def _sq_attackers(board: chess.Board, sq: int, occ: int) -> int:
    """Bitboard of all pieces that attack *sq* under occupancy *occ*."""
    attackers  = chess.BB_KNIGHT_ATTACKS[sq] & int(board.knights)
    attackers |= chess.BB_KING_ATTACKS[sq]   & int(board.kings)
    # Pawns: white pawns sitting on black-pawn-attack squares (and vice-versa)
    attackers |= (chess.BB_PAWN_ATTACKS[chess.BLACK][sq]
                  & int(board.pawns) & int(board.occupied_co[chess.WHITE]))
    attackers |= (chess.BB_PAWN_ATTACKS[chess.WHITE][sq]
                  & int(board.pawns) & int(board.occupied_co[chess.BLACK]))
    # Sliding pieces with *current* occupancy (handles x-rays after removal)
    attackers |= (_sliding_attacks(sq, occ, rook=True, bishop=False)
                  & (int(board.rooks) | int(board.queens)))
    attackers |= (_sliding_attacks(sq, occ, rook=False, bishop=True)
                  & (int(board.bishops) | int(board.queens)))
    return attackers & occ


# Piece values for SEE (king = very large so king-captures are always "winning"
# but losing the king on the next recapture makes the exchange terrible).
_SEE_VALUES = [0, 100, 320, 330, 500, 900, 20_000]  # indexed by piece type


def see(board: chess.Board, move: chess.Move) -> int:
    """Return the estimated material gain of *move* after all recaptures.

    Positive  = the capture sequence wins material for the side to move.
    Negative  = the capture loses material (e.g. QxP defended by a knight).
    """
    to_sq = move.to_square
    from_sq = move.from_square

    victim_pt = board.piece_type_at(to_sq)
    if victim_pt is None:
        if board.is_en_passant(move):
            victim_pt = chess.PAWN
        else:
            return 0

    attacker_pt = board.piece_type_at(from_sq)

    # Build the swap list
    gains = [0] * 33

    if move.promotion:
        gains[0] = (_SEE_VALUES[victim_pt]
                    + _SEE_VALUES[move.promotion] - _SEE_VALUES[chess.PAWN])
        cur_val = _SEE_VALUES[move.promotion]
    else:
        gains[0] = _SEE_VALUES[victim_pt]
        cur_val = _SEE_VALUES[attacker_pt]

    occ = int(board.occupied) ^ (1 << from_sq)
    if board.is_en_passant(move):
        ep_sq = to_sq + (-8 if board.turn == chess.WHITE else 8)
        occ ^= (1 << ep_sq)

    attadef = _sq_attackers(board, to_sq, occ)
    side = not board.turn  # opponent recaptures first
    d = 0

    while True:
        # Check for a recapturer *before* writing a speculative gain entry.
        side_att = attadef & int(board.occupied_co[side])
        if not side_att:
            break

        d += 1
        gains[d] = cur_val - gains[d - 1]

        # If both stand-pat and capturing are losing, stop early
        if max(-gains[d - 1], gains[d]) < 0:
            break

        # Least-valuable attacker of *side*
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                   chess.ROOK, chess.QUEEN, chess.KING):
            pt_bb = side_att & int(board.pieces(pt, side))
            if pt_bb:
                sq = chess.lsb(pt_bb)
                cur_val = _SEE_VALUES[pt]
                # Pawn promoting on capture
                if pt == chess.PAWN and chess.square_rank(to_sq) in (0, 7):
                    gains[d] += _SEE_VALUES[chess.QUEEN] - _SEE_VALUES[chess.PAWN]
                    cur_val = _SEE_VALUES[chess.QUEEN]
                occ ^= (1 << sq)
                attadef = _sq_attackers(board, to_sq, occ)
                break

        side = not side

    # Negamax: at each ply the side to move may choose not to recapture
    while d > 0:
        d -= 1
        gains[d] = -max(-gains[d], gains[d + 1])

    return gains[0]


# ---------------------------------------------------------------------------
# Late Move Reduction table (precomputed)
# ---------------------------------------------------------------------------
# Reductions scale logarithmically with both depth and move index.
# Later moves at deeper depths get bigger reductions.

_LMR_TABLE = [[0] * 64 for _ in range(MAX_DEPTH + 1)]
for _d in range(1, MAX_DEPTH + 1):
    for _m in range(1, 64):
        _LMR_TABLE[_d][_m] = max(1, int(0.5 + math.log(_d) * math.log(_m) / 2.25))

# Late move pruning thresholds by depth: skip quiet moves past this index.
_LMP_THRESHOLD = [0] + [3 + d * 3 for d in range(1, MAX_DEPTH + 1)]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class Searcher:
    def __init__(self, info_handler=None):
        self.tt = TranspositionTable()
        self.orderer = MoveOrderer()
        self.nodes = 0
        self.stop_event = threading.Event()
        self.best_move: chess.Move | None = None
        self.start_time = 0.0
        self.time_limit: float | None = None
        # Optional callback for search info lines. Defaults to print() for UCI.
        self._info_handler = info_handler or (lambda msg: print(msg, flush=True))

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

            # SEE pruning: skip captures that lose material
            if see(board, move) < 0:
                continue

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

        # Static eval (computed once, reused by pruning heuristics below)
        static_eval = evaluate(board) if not in_check else None

        # Reverse futility pruning: at low depths, if our position is already
        # so good that even a conservative margin can't drop below beta, prune.
        if (not is_pv and not in_check and depth <= 3 and not is_root
                and static_eval is not None
                and static_eval - 120 * depth >= beta
                and abs(beta) < MATE_SCORE - MAX_DEPTH):
            return static_eval

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

        # Futility pruning margin: at low depths, quiet moves that can't
        # possibly raise alpha are skipped in the move loop below.
        futile = False
        if (not is_pv and not in_check and depth <= 2
                and static_eval is not None
                and static_eval + 150 * depth <= alpha
                and abs(alpha) < MATE_SCORE - MAX_DEPTH):
            futile = True

        moves = self.orderer.ordered_moves(board, ply, tt_move)

        if not moves:
            if in_check:
                return -MATE_SCORE + ply
            return 0  # stalemate

        best_score = -INF
        best_move = moves[0]
        flag = TTFlag.ALPHA

        for i, move in enumerate(moves):
            is_quiet = (not board.is_capture(move)
                        and not move.promotion)

            # Futility pruning: skip quiet moves that can't raise alpha
            if (futile and i > 0 and is_quiet
                    and not board.gives_check(move)):
                continue

            # Late move pruning: at low depths, skip very late quiet moves
            if (not is_pv and not in_check and is_quiet
                    and depth <= 3 and i >= _LMP_THRESHOLD[depth]
                    and not board.gives_check(move)):
                continue

            # Check extension
            ext = 1 if board.gives_check(move) else 0

            board.push(move)

            # Late Move Reductions (logarithmic)
            if (i >= 3 and depth >= 2 and not in_check
                    and is_quiet and not board.is_check()):
                R = _LMR_TABLE[min(depth, MAX_DEPTH)][min(i, 63)]
                # Don't reduce below depth 1
                reduced_depth = max(depth - 1 - R + ext, 1)
                score = -self._alpha_beta(board, reduced_depth, -alpha - 1, -alpha, ply + 1)
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
        prev_score = 0
        ASP_WINDOW = 50

        for depth in range(1, max_depth + 1):
            # Aspiration windows: use a narrow window around the previous
            # iteration's score.  On fail-high or fail-low, widen and retry.
            # Depth 1-3 use a full window (not enough info to guess yet).
            if depth <= 3:
                alpha, beta = -INF, INF
            else:
                alpha = prev_score - ASP_WINDOW
                beta = prev_score + ASP_WINDOW

            while True:
                score = self._alpha_beta(board, depth, alpha, beta, 0)

                if self.stop_event.is_set():
                    break

                if score <= alpha:
                    # Fail-low: widen downward
                    alpha = max(alpha - ASP_WINDOW * 4, -INF)
                elif score >= beta:
                    # Fail-high: widen upward
                    beta = min(beta + ASP_WINDOW * 4, INF)
                else:
                    break  # score is inside the window

            if self.stop_event.is_set():
                break

            prev_score = score
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

            self._info_handler(
                f"info depth {depth} {score_str} nodes {self.nodes} "
                f"nps {nps} time {int(elapsed * 1000)} "
                f"pv {best_for_iter.uci() if best_for_iter else '(none)'}"
            )

            # If we found a forced mate, stop searching deeper
            if abs(score) >= MATE_SCORE - MAX_DEPTH:
                break

        return best_for_iter or list(board.legal_moves)[0]


# ---------------------------------------------------------------------------
# UCI protocol
# ---------------------------------------------------------------------------

