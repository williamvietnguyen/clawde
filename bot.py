#!/usr/bin/env python3
"""Discord bot for playing chess against the Clawde engine."""

import os
import asyncio
import functools
import logging
from dataclasses import dataclass, field
from enum import Enum

import chess
import discord
from discord import app_commands

from engine import Searcher

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENGINE_TIME_LIMIT = 5.0  # seconds per move (tune for your hardware)
BOT_TOKEN_ENV = "CLAWDE_BOT_TOKEN"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("clawde")

# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------


@dataclass
class GameState:
    board: chess.Board = field(default_factory=chess.Board)
    searcher: Searcher = field(default_factory=Searcher)
    player_color: chess.Color = chess.WHITE
    player_id: int = 0
    move_history: list[str] = field(default_factory=list)


# guild_id -> GameState
games: dict[int, GameState] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def render_board(board: chess.Board, orientation: chess.Color = chess.WHITE) -> str:
    return f"```\n{board.unicode(borders=True, orientation=orientation)}\n```"


def parse_move(board: chess.Board, move_str: str) -> chess.Move | None:
    # Try SAN first (e4, Nf3, O-O, e8=Q)
    try:
        return board.parse_san(move_str)
    except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
        pass
    # Try UCI (e2e4, g1f3, e7e8q)
    try:
        move = chess.Move.from_uci(move_str)
        if move in board.legal_moves:
            return move
    except chess.InvalidMoveError:
        pass
    return None


def format_move_list(moves: list[str]) -> str:
    lines = []
    for i in range(0, len(moves), 2):
        num = i // 2 + 1
        white = moves[i]
        black = moves[i + 1] if i + 1 < len(moves) else ""
        lines.append(f"{num}. {white} {black}".rstrip())
    return "\n".join(lines) if lines else "(no moves yet)"


def check_game_over(board: chess.Board) -> str | None:
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        return f"**Checkmate!** {winner} wins."
    if board.is_stalemate():
        return "**Stalemate!** The game is a draw."
    if board.is_insufficient_material():
        return "**Draw** by insufficient material."
    if board.can_claim_fifty_moves():
        return "**Draw** by the fifty-move rule."
    if board.is_fivefold_repetition():
        return "**Draw** by fivefold repetition."
    if board.is_repetition(3):
        return "**Draw** by threefold repetition."
    return None


def legal_moves_hint(board: chess.Board, limit: int = 15) -> str:
    sans = [board.san(m) for m in board.legal_moves]
    if len(sans) <= limit:
        return ", ".join(f"`{s}`" for s in sans)
    return ", ".join(f"`{s}`" for s in sans[:limit]) + ", ..."


async def run_engine(searcher: Searcher, board: chess.Board) -> chess.Move:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(searcher.search, board, time_limit=ENGINE_TIME_LIMIT),
    )


# ---------------------------------------------------------------------------
# Bot setup
# ---------------------------------------------------------------------------

intents = discord.Intents.default()
bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)


class ColorChoice(Enum):
    white = "white"
    black = "black"


def require_guild(interaction: discord.Interaction) -> int | None:
    return interaction.guild_id


# ---------------------------------------------------------------------------
# /play
# ---------------------------------------------------------------------------


@tree.command(name="play", description="Start a new chess game against Clawde")
@app_commands.describe(color="Choose your color (default: white)")
async def play_command(
    interaction: discord.Interaction, color: ColorChoice = ColorChoice.white
):
    guild_id = interaction.guild_id
    if guild_id is None:
        await interaction.response.send_message(
            "This command can only be used in a server."
        )
        return

    if guild_id in games:
        await interaction.response.send_message(
            "A game is already in progress! Use `/resign` to end it first."
        )
        return

    player_color = chess.WHITE if color == ColorChoice.white else chess.BLACK
    game = GameState(player_color=player_color, player_id=interaction.user.id)
    games[guild_id] = game

    if player_color == chess.WHITE:
        msg = (
            f"**New game!** You are **White**.\n"
            f"{render_board(game.board, chess.WHITE)}\n"
            f"Your turn — use `/move` to play."
        )
        await interaction.response.send_message(msg)
    else:
        await interaction.response.defer()
        try:
            engine_result = await run_engine(game.searcher, game.board)
        except Exception as e:
            del games[guild_id]
            await interaction.followup.send(f"Engine error: {e}")
            return

        engine_san = game.board.san(engine_result)
        game.board.push(engine_result)
        game.move_history.append(engine_san)

        msg = (
            f"**New game!** You are **Black**.\n"
            f"Clawde opens with **{engine_san}**\n"
            f"{render_board(game.board, chess.BLACK)}\n"
            f"Your turn!"
        )
        await interaction.followup.send(msg)


# ---------------------------------------------------------------------------
# /move
# ---------------------------------------------------------------------------


@tree.command(name="move", description="Make a chess move")
@app_commands.describe(
    move="Your move in algebraic (e4, Nf3, O-O) or UCI (e2e4) notation"
)
async def move_command(interaction: discord.Interaction, move: str):
    guild_id = interaction.guild_id
    if guild_id is None:
        await interaction.response.send_message(
            "This command can only be used in a server."
        )
        return

    game = games.get(guild_id)
    if game is None:
        await interaction.response.send_message(
            "No active game. Use `/play` to start one."
        )
        return

    if interaction.user.id != game.player_id:
        await interaction.response.send_message(
            "Only the player who started this game can make moves.", ephemeral=True
        )
        return

    if game.board.turn != game.player_color:
        await interaction.response.send_message(
            "It's not your turn — something went wrong. Try `/resign` and start over."
        )
        return

    parsed = parse_move(game.board, move)
    if parsed is None:
        await interaction.response.send_message(
            f"Invalid move: `{move}`\n"
            f"Legal moves: {legal_moves_hint(game.board)}",
            ephemeral=True,
        )
        return

    # Push player's move
    san = game.board.san(parsed)
    game.board.push(parsed)
    game.move_history.append(san)

    # Check game over after player's move
    result = check_game_over(game.board)
    if result:
        board_str = render_board(game.board, game.player_color)
        del games[guild_id]
        await interaction.response.send_message(
            f"You played **{san}**\n{result}\n{board_str}"
        )
        return

    # Engine thinks
    await interaction.response.defer()
    try:
        engine_result = await run_engine(game.searcher, game.board)
    except Exception as e:
        # Undo player's move so the game isn't in a broken state
        game.board.pop()
        game.move_history.pop()
        await interaction.followup.send(f"Engine error: {e}")
        return

    engine_san = game.board.san(engine_result)
    game.board.push(engine_result)
    game.move_history.append(engine_san)

    # Check game over after engine's move
    result = check_game_over(game.board)
    board_str = render_board(game.board, game.player_color)

    if result:
        del games[guild_id]
        await interaction.followup.send(
            f"You played **{san}** — Clawde responds **{engine_san}**\n"
            f"{result}\n{board_str}"
        )
        return

    check_str = " **Check!**" if game.board.is_check() else ""
    await interaction.followup.send(
        f"You played **{san}** — Clawde responds **{engine_san}**{check_str}\n"
        f"{board_str}\n"
        f"Your turn!"
    )


# ---------------------------------------------------------------------------
# /board
# ---------------------------------------------------------------------------


@tree.command(name="board", description="Show the current board")
async def board_command(interaction: discord.Interaction):
    guild_id = interaction.guild_id
    if guild_id is None:
        await interaction.response.send_message(
            "This command can only be used in a server."
        )
        return

    game = games.get(guild_id)
    if game is None:
        await interaction.response.send_message(
            "No active game. Use `/play` to start one."
        )
        return

    turn = "Your turn" if game.board.turn == game.player_color else "Clawde is thinking..."
    move_num = game.board.fullmove_number
    await interaction.response.send_message(
        f"**Move {move_num}** — {turn}\n"
        f"{render_board(game.board, game.player_color)}"
    )


# ---------------------------------------------------------------------------
# /resign
# ---------------------------------------------------------------------------


@tree.command(name="resign", description="Resign the current game")
async def resign_command(interaction: discord.Interaction):
    guild_id = interaction.guild_id
    if guild_id is None:
        await interaction.response.send_message(
            "This command can only be used in a server."
        )
        return

    game = games.get(guild_id)
    if game is None:
        await interaction.response.send_message("No active game to resign.")
        return

    if interaction.user.id != game.player_id:
        await interaction.response.send_message(
            "Only the player who started this game can resign.", ephemeral=True
        )
        return

    color = "White" if game.player_color == chess.WHITE else "Black"
    del games[guild_id]
    await interaction.response.send_message(
        f"**{interaction.user.display_name}** ({color}) resigns. Clawde wins!"
    )


# ---------------------------------------------------------------------------
# /moves
# ---------------------------------------------------------------------------


@tree.command(name="moves", description="Show the move history")
async def moves_command(interaction: discord.Interaction):
    guild_id = interaction.guild_id
    if guild_id is None:
        await interaction.response.send_message(
            "This command can only be used in a server."
        )
        return

    game = games.get(guild_id)
    if game is None:
        await interaction.response.send_message("No active game.")
        return

    history = format_move_list(game.move_history)
    await interaction.response.send_message(f"```\n{history}\n```")


# ---------------------------------------------------------------------------
# /fen
# ---------------------------------------------------------------------------


@tree.command(name="fen", description="Show the current position as a FEN string")
async def fen_command(interaction: discord.Interaction):
    guild_id = interaction.guild_id
    if guild_id is None:
        await interaction.response.send_message(
            "This command can only be used in a server."
        )
        return

    game = games.get(guild_id)
    if game is None:
        await interaction.response.send_message("No active game.")
        return

    await interaction.response.send_message(f"```\n{game.board.fen()}\n```")


# ---------------------------------------------------------------------------
# Bot events
# ---------------------------------------------------------------------------


@bot.event
async def on_ready():
    await tree.sync()
    log.info(f"Clawde bot online as {bot.user}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    token = os.environ.get(BOT_TOKEN_ENV)
    if not token:
        print(f"Error: set the {BOT_TOKEN_ENV} environment variable.")
        print(f"  export {BOT_TOKEN_ENV}='your-bot-token-here'")
        raise SystemExit(1)
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
