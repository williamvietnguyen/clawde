# Clawde

A UCI-compatible chess engine written in Python, with a Discord bot for playing against it.

## Files

- `engine.py` — Chess engine with iterative deepening alpha-beta search, quiescence search, transposition table, piece-square table evaluation, and full UCI protocol support
- `bot.py` — Discord bot that lets you play against the engine via slash commands

## Slash Commands

| Command | What it does |
|---------|-------------|
| `/play` | Start a game (choose white or black) |
| `/move` | Make a move (`e4`, `Nf3`, `O-O`, or UCI like `e2e4`) |
| `/board` | Show the current board |
| `/resign` | Resign the game |
| `/moves` | Show move history |
| `/fen` | Show the FEN string |

## Setup

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install chess discord.py
```

### 2. Create a Discord bot

1. Go to https://discord.com/developers/applications
2. Create a new application
3. Go to **Bot** and copy the token
4. Go to **OAuth2 → URL Generator**, select the `bot` and `applications.commands` scopes
5. Use the generated URL to invite the bot to your server

### 3. Run the bot

```bash
source venv/bin/activate
export CLAWDE_BOT_TOKEN='your-token-here'
python bot.py
```

### 4. Run on a Raspberry Pi with systemd (optional)

Create `/etc/systemd/system/clawde.service`:

```ini
[Unit]
Description=Clawde Chess Bot
After=network.target

[Service]
ExecStart=/path/to/venv/bin/python /path/to/bot.py
WorkingDirectory=/path/to/clawde
Environment=CLAWDE_BOT_TOKEN=your-token-here
Restart=on-failure
User=pi

[Install]
WantedBy=multi-user.target
```

Then enable and start it:

```bash
sudo systemctl enable clawde
sudo systemctl start clawde
```

## Configuration

The engine thinks for 5 seconds per move by default. To tune this for your hardware, edit the `ENGINE_TIME_LIMIT` constant at the top of `bot.py`.

## UCI mode

The engine also works standalone with any UCI-compatible chess GUI (Arena, CuteChess, etc.):

```bash
source venv/bin/activate
python engine.py
```
