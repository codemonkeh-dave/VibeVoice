#!/usr/bin/env python3
"""VibeVoice TTS server — loads model once, serves audio over HTTP with streaming."""

import copy
import os
import struct
import threading
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.modular.streamer import AudioStreamer
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

VOICES_DIR = os.path.join(os.path.dirname(__file__), "demo", "voices", "streaming_model")
MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"
PORT = 8723
SAMPLE_RATE = 24000

app = FastAPI()
state = {}


class SayRequest(BaseModel):
    text: str
    voice: str = "en-Carter_man"
    cfg_scale: float = 1.5


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_voice(name: str) -> str:
    path = os.path.join(VOICES_DIR, f"{name}.pt")
    if os.path.exists(path):
        return path
    for f in os.listdir(VOICES_DIR):
        if f.lower().startswith(name.lower()):
            return os.path.join(VOICES_DIR, f)
    raise FileNotFoundError(f"Voice '{name}' not found")


@app.on_event("startup")
def startup():
    device = get_device()
    print(f"Device: {device}")
    print(f"Loading model: {MODEL_ID}")

    if device == "mps":
        dtype, attn = torch.float32, "sdpa"
    elif device == "cuda":
        dtype, attn = torch.bfloat16, "flash_attention_2"
    else:
        dtype, attn = torch.float32, "sdpa"

    processor = VibeVoiceStreamingProcessor.from_pretrained(MODEL_ID)

    try:
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            MODEL_ID, dtype=dtype, attn_implementation=attn,
            device_map=None if device == "mps" else device,
        )
        if device == "mps":
            model.to("mps")
    except Exception:
        if attn == "flash_attention_2":
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                MODEL_ID, dtype=dtype, attn_implementation="sdpa",
                device_map=None if device == "mps" else device,
            )
            if device == "mps":
                model.to("mps")
        else:
            raise

    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)

    state["processor"] = processor
    state["model"] = model
    state["device"] = device
    print(f"Model loaded, server ready on port {PORT}")


@app.get("/health")
def health():
    return {"status": "ok", "device": state.get("device", "loading")}


@app.get("/voices")
def voices():
    names = sorted(
        os.path.splitext(f)[0] for f in os.listdir(VOICES_DIR) if f.endswith(".pt")
    )
    return {"voices": names}


@app.post("/shutdown")
def shutdown():
    import signal
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "shutting down"}


@app.post("/say")
def say(req: SayRequest):
    """Non-streaming endpoint — returns complete WAV."""
    processor = state["processor"]
    model = state["model"]
    device = state["device"]

    voice_path = resolve_voice(req.voice)
    voice_data = torch.load(voice_path, map_location=device, weights_only=False)

    text = req.text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')

    inputs = processor.process_input_with_cached_prompt(
        text=text, cached_prompt=voice_data,
        padding=True, return_tensors="pt", return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    start = time.time()
    outputs = model.generate(
        **inputs, max_new_tokens=None, cfg_scale=req.cfg_scale,
        tokenizer=processor.tokenizer, generation_config={"do_sample": False},
        verbose=False, all_prefilled_outputs=copy.deepcopy(voice_data),
    )
    gen_time = time.time() - start

    audio = outputs.speech_outputs[0]
    audio_dur = audio.shape[-1] / SAMPLE_RATE
    rtf = gen_time / audio_dur if audio_dur > 0 else float("inf")
    print(f"Generated {audio_dur:.1f}s audio in {gen_time:.1f}s (RTF: {rtf:.2f}x)")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    processor.save_audio(audio, output_path=tmp)
    with open(tmp, "rb") as f:
        wav_bytes = f.read()
    os.unlink(tmp)

    return Response(
        content=wav_bytes, media_type="audio/wav",
        headers={
            "X-Audio-Duration": f"{audio_dur:.2f}",
            "X-Generation-Time": f"{gen_time:.2f}",
            "X-RTF": f"{rtf:.2f}",
        },
    )


@app.post("/stream")
def stream(req: SayRequest):
    """Streaming endpoint — yields raw PCM16 chunks as they're generated."""
    model = state["model"]
    processor = state["processor"]
    device = state["device"]

    voice_path = resolve_voice(req.voice)
    voice_data = torch.load(voice_path, map_location=device, weights_only=False)

    text = req.text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')

    inputs = processor.process_input_with_cached_prompt(
        text=text, cached_prompt=voice_data,
        padding=True, return_tensors="pt", return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
    errors = []

    def generate_thread():
        try:
            model.generate(
                **inputs, max_new_tokens=None, cfg_scale=req.cfg_scale,
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                audio_streamer=audio_streamer,
                all_prefilled_outputs=copy.deepcopy(voice_data),
            )
        except Exception as e:
            errors.append(e)
            audio_streamer.end()

    thread = threading.Thread(target=generate_thread, daemon=True)
    thread.start()

    def chunk_generator():
        stream = audio_streamer.get_stream(0)
        for audio_chunk in stream:
            if torch.is_tensor(audio_chunk):
                audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
            else:
                audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
            if audio_chunk.ndim > 1:
                audio_chunk = audio_chunk.reshape(-1)
            peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
            if peak > 1.0:
                audio_chunk = audio_chunk / peak
            # Convert float32 -> PCM16 bytes
            pcm = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)
            yield pcm.tobytes()
        thread.join()

    return StreamingResponse(
        chunk_generator(),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(SAMPLE_RATE), "X-Channels": "1", "X-Sample-Width": "2"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
