#!/usr/bin/env python3
"""Say text aloud using VibeVoice. Starts the server automatically if needed."""

import argparse
import numpy as np
import os
import signal
import subprocess
import sys
import time
import threading

import requests
import sounddevice as sd

SERVER_URL = "http://127.0.0.1:8723"
SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "say_server.py")
VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
PID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".say_server.pid")
SAMPLE_RATE = 24000


def server_running():
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=2)
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def start_server():
    python = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "say_server.log")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [python, SERVER_SCRIPT],
        stdout=log_file, stderr=log_file,
        start_new_session=True,
    )
    # Save PID for later shutdown
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))
    print(f"Starting server (pid: {proc.pid}, log: {log_path})...")
    for i in range(120):
        time.sleep(1)
        if server_running():
            print("Server ready.")
            return
        if i % 10 == 9:
            print(f"  still loading model ({i+1}s)...")
    print("Server failed to start. Check say_server.log")
    sys.exit(1)


def stop_server():
    # Try HTTP shutdown first
    if server_running():
        try:
            requests.post(f"{SERVER_URL}/shutdown", timeout=5)
            print("Server stopped.")
            _cleanup_pid()
            return
        except Exception:
            pass
    # Fall back to PID file
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Server stopped (pid: {pid}).")
        except ProcessLookupError:
            print("Server was not running.")
        _cleanup_pid()
    else:
        print("Server not running.")


def _cleanup_pid():
    if os.path.exists(PID_FILE):
        os.unlink(PID_FILE)


def ensure_server():
    if not server_running():
        start_server()


def list_voices():
    ensure_server()
    r = requests.get(f"{SERVER_URL}/voices")
    for v in r.json()["voices"]:
        print(f"  {v}")


def say(text, voice, cfg):
    ensure_server()

    r = requests.post(f"{SERVER_URL}/stream", json={
        "text": text, "voice": voice, "cfg_scale": cfg,
    }, stream=True, timeout=300)

    if r.status_code != 200:
        print(f"Error: {r.status_code} {r.text}")
        sys.exit(1)

    # Stream audio playback — play chunks as they arrive from the server
    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    stream.start()

    total_samples = 0
    start = time.time()
    first_chunk_time = None
    buf = b""

    for chunk in r.iter_content(chunk_size=4096):
        if not chunk:
            continue
        buf += chunk
        # sounddevice needs complete samples (2 bytes each for int16)
        usable = len(buf) - (len(buf) % 2)
        if usable == 0:
            continue
        pcm = np.frombuffer(buf[:usable], dtype=np.int16).reshape(-1, 1)
        buf = buf[usable:]
        stream.write(pcm)
        total_samples += len(pcm)
        if first_chunk_time is None:
            first_chunk_time = time.time() - start
            print(f"First audio at {first_chunk_time:.2f}s")

    # Play any remaining bytes
    if len(buf) >= 2:
        usable = len(buf) - (len(buf) % 2)
        pcm = np.frombuffer(buf[:usable], dtype=np.int16).reshape(-1, 1)
        stream.write(pcm)
        total_samples += len(pcm)

    gen_time = time.time() - start
    audio_dur = total_samples / SAMPLE_RATE

    # Wait for playback to finish (stream.write is non-blocking for buffered data)
    remaining = audio_dur - gen_time
    if remaining > 0:
        time.sleep(remaining + 0.1)

    stream.stop()
    stream.close()
    print(f"Played {audio_dur:.1f}s of audio (generated in {gen_time:.1f}s, RTF: {gen_time/audio_dur:.2f}x)")



def main():
    parser = argparse.ArgumentParser(description="Say text using VibeVoice")
    parser.add_argument("text", nargs="*", help="Text to speak")
    parser.add_argument("--voice", "-v", default="en-Carter_man", help="Voice name (default: en-Carter_man)")
    parser.add_argument("--list-voices", "-l", action="store_true", help="List available voices")
    parser.add_argument("--cfg", type=float, default=1.5, help="CFG scale (default: 1.5)")
    parser.add_argument("--server", action="store_true", help="Just start the server, don't say anything")
    parser.add_argument("--stop", action="store_true", help="Stop the running server")
    args = parser.parse_args()

    if args.stop:
        stop_server()
        return

    if args.server:
        ensure_server()
        print(f"Server running at {SERVER_URL}")
        return

    if args.list_voices:
        print("Available voices:")
        list_voices()
        return

    text = " ".join(args.text) if args.text else None
    if not text:
        text = sys.stdin.read().strip()
    if not text:
        parser.error("No text provided. Pass text as arguments or pipe via stdin.")

    say(text, args.voice, args.cfg)


if __name__ == "__main__":
    main()
