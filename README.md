# Clawde

A UCI-compatible chess engine written in Python, with a Lichess bot for playing live games on lichess.org.

## Files

- `engine.py` — Chess engine with iterative deepening alpha-beta search, quiescence search, transposition table, piece-square table evaluation, and full UCI protocol support
- `lichess_bot.py` — Lichess bot that connects the engine to play live games via the Lichess Bot API

## Setup

### 1. Install dependencies

Requires Python 3.12+.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Create a Lichess BOT account

1. Create a new Lichess account (or use an existing one with no rated games)
2. Upgrade it to a BOT account via the [Lichess API](https://lichess.org/api#tag/Bot/operation/botAccountUpgrade)
3. Generate a personal API token at https://lichess.org/account/oauth/token with the `bot:play` scope

### 3. Run the bot

```bash
source venv/bin/activate
export LICHESS_BOT_TOKEN='lip_...'
python lichess_bot.py
```

The bot will connect, log "Listening for events", and accept incoming challenges automatically.

### 4. Deploy on a Jetson Nano with systemd (optional)

Create `/etc/systemd/system/clawde.service`:

```ini
[Unit]
Description=Clawde Chess Bot
After=network.target

[Service]
ExecStart=/path/to/venv/bin/python /path/to/lichess_bot.py
WorkingDirectory=/path/to/clawde
Environment=LICHESS_BOT_TOKEN=lip_your_token_here
Restart=on-failure
User=jetson

[Install]
WantedBy=multi-user.target
```

Then enable and start it:

```bash
sudo systemctl enable clawde
sudo systemctl start clawde
```

## UCI mode

The engine also works standalone with any UCI-compatible chess GUI (Arena, CuteChess, etc.):

```bash
source venv/bin/activate
python engine.py
```
