# say.py

A command-line tool for text-to-speech using Microsoft's [VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) model.

A background server keeps the model loaded in memory so subsequent calls are fast — only the first invocation pays the model loading cost.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
# Say something (starts server automatically on first run)
./say.py "Hello, this is VibeVoice."

# Choose a voice
./say.py -v en-Emma_woman "Hello from Emma."

# Pipe text in
echo "Some text to speak" | ./say.py

# List available voices
./say.py --list-voices

# Pre-start the server without saying anything
./say.py --server

# Stop the server
./say.py --stop
```

## Options

| Flag | Description |
|------|-------------|
| `-v`, `--voice` | Voice preset name (default: `en-Carter_man`) |
| `-l`, `--list-voices` | List all available voice presets |
| `--cfg` | CFG scale for generation (default: 1.5) |
| `--server` | Start the server without generating speech |
| `--stop` | Stop the running server |

## Architecture

- **`say.py`** — thin client that sends text to the server and plays the returned audio via `afplay` (macOS) or `paplay`/`aplay` (Linux)
- **`say_server.py`** — FastAPI server on `127.0.0.1:8723` that loads the model once at startup and serves a `/say` endpoint returning WAV audio

Server logs are written to `say_server.log`.

## Voices

English voices: Carter, Davis, Emma, Frank, Grace, Mike. Additional experimental voices (German, French, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish) can be downloaded with:

```bash
bash demo/download_experimental_voices.sh
```
