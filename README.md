<p align="center">
  <img src="assets/logo.svg" alt="OwlHearYou logo" width="200"/>
</p>

# OwlHearYou

Real-time speech-to-text and text-to-speech service over WebSockets. Supports two STT backends and a Kokoro TTS synthesizer, all selectable via environment variables.

## Setup

See [INSTALL.md](INSTALL.md) for full setup instructions.

## Running

### Server (podman-compose)

```bash
# Copy the example and adjust image tag / target as needed
cp podman-compose.example.yml podman-compose.yml

# Build the image
podman-compose build

# Start the server
podman-compose up
```

The server exposes:
- `ws://localhost:8000/ws/transcribe` — real-time STT
- `ws://localhost:8000/ws/synthesize` — streaming TTS
- `http://localhost:8000/health` — health check

### Client (host)

```bash
# Stream from microphone for server-side transcription (STT)
python -m src.client

# Synthesize text to speech (TTS)
python -m src.client --mode tts --text "Hello, world!"

# Use a specific TTS voice
python -m src.client --mode tts --text "Hello!" --voice af_sarah

# Point at a different server
python -m src.client --uri ws://somehost:8000

# List available audio input devices (useful if audio fails to open)
python -m src.client --list-devices
```

If the client fails to open the microphone, it will print the available input devices and a hint. You can pin a specific device by index:

```bash
AUDIO_DEVICE=9 python -m src.client
```

### Server (local, without container)

```bash
# Nemotron (default, GPU)
./scripts/run.sh

# Nemotron on CPU
STT_DEVICE=cpu ./scripts/run.sh

# Whisper on GPU
STT_BACKEND=whisper ./scripts/run.sh

# Whisper on CPU
STT_BACKEND=whisper STT_DEVICE=cpu ./scripts/run.sh

# Whisper with a specific model and language
STT_BACKEND=whisper WHISPER_MODEL=large-v3 WHISPER_LANGUAGE=de ./scripts/run.sh
```

## Environment variables

| Variable | Values | Default |
|---|---|---|
| `STT_BACKEND` | `nemotron`, `whisper` | `nemotron` |
| `STT_DEVICE` | `cuda`, `cpu`, `auto` | `cuda` |
| `WHISPER_MODEL` | any faster-whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`, …) | `base` |
| `WHISPER_LANGUAGE` | BCP-47 language code (`en`, `de`, `fr`, …) or unset for auto-detect | unset |
| `KOKORO_VOICE` | any voice name from the Kokoro voices file (e.g. `af_heart`, `af_sarah`) | `af_heart` |

`STT_DEVICE=auto` lets faster-whisper pick the best available device automatically; for Nemotron it behaves the same as `cuda`.

## Backends

### STT: Nemotron (default)

Uses `sherpa-onnx` with an NVIDIA Nemotron transducer model. Processes audio chunk-by-chunk as it arrives and emits partial hypotheses in real time — words appear as you speak. Endpoint detection triggers a final result after a configurable silence window.

- **Latency:** very low — partial results every ~30ms
- **Concurrency:** fully parallel — each connection has its own independent stream, no shared state

### STT: Whisper

Uses `faster-whisper` (CTranslate2 backend). Buffers audio per-connection and transcribes on silence — a complete sentence appears after the speaker pauses. No partial hypotheses.

- **Latency:** higher — output appears only after a silence endpoint (~1.3s of quiet)
- **Concurrency:** serialized — `WhisperModel.transcribe()` is not thread-safe, so a semaphore limits the model to one transcription at a time across all connections. Audio buffering and WebSocket I/O remain concurrent; only the transcription step queues.

#### Making Whisper fully concurrent

For production multi-user deployments there are two options:

1. **Multiple model instances** — instantiate one `WhisperModel` per worker process (via `uvicorn --workers N`). Each process owns its model and has no contention. Cost: N × VRAM.

2. **Dedicated inference server** — run [Triton Inference Server](https://github.com/triton-inference-server/server) or [whisper.cpp server](https://github.com/ggerganov/whisper.cpp) as a separate service. These handle dynamic batching internally, serving many concurrent requests from a single model instance efficiently. The transcriber session becomes a thin HTTP/gRPC client.

### TTS: Kokoro

Uses `kokoro-onnx` with the official `kokoro-v1.0.onnx` model. On first startup the server downloads the model (~310 MB) and voices file automatically to `models/kokoro/`.

The `/ws/synthesize` endpoint accepts text messages and streams back raw float32 PCM at 24 kHz mono. Each utterance ends with an empty bytes sentinel (`b""`). A voice can be selected per-connection via query parameter:

```
ws://localhost:8000/ws/synthesize?voice=af_sarah
```

If omitted, the server uses the `KOKORO_VOICE` environment variable (default: `af_heart`).
