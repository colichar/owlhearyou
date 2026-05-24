# OwlHearYou

Real-time speech-to-text service over WebSockets. Supports two transcription backends selectable via environment variable.

## Setup

See [INSTALL.md](INSTALL.md) for full setup instructions.

## Running

```bash
# Nemotron (default, GPU)
./run.sh

# Nemotron on CPU
STT_DEVICE=cpu ./run.sh

# Whisper on GPU
STT_BACKEND=whisper ./run.sh

# Whisper on CPU
STT_BACKEND=whisper STT_DEVICE=cpu ./run.sh

# Whisper with a specific model and language
STT_BACKEND=whisper WHISPER_MODEL=large-v3 WHISPER_LANGUAGE=de ./run.sh
```

## Environment variables

| Variable | Values | Default |
|---|---|---|
| `STT_BACKEND` | `nemotron`, `whisper` | `nemotron` |
| `STT_DEVICE` | `cuda`, `cpu`, `auto` | `cuda` |
| `WHISPER_MODEL` | any faster-whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`, …) | `base` |
| `WHISPER_LANGUAGE` | BCP-47 language code (`en`, `de`, `fr`, …) or unset for auto-detect | unset |

`STT_DEVICE=auto` lets faster-whisper pick the best available device automatically; for Nemotron it behaves the same as `cuda`.

## Backends

### Nemotron (default)

Uses `sherpa-onnx` with an NVIDIA Nemotron transducer model. Processes audio chunk-by-chunk as it arrives and emits partial hypotheses in real time — words appear as you speak. Endpoint detection triggers a final result after a configurable silence window.

- **Latency:** very low — partial results every ~30ms
- **Concurrency:** fully parallel — each connection has its own independent stream, no shared state

### Whisper

Uses `faster-whisper` (CTranslate2 backend). Buffers audio per-connection and transcribes on silence — a complete sentence appears after the speaker pauses. No partial hypotheses.

- **Latency:** higher — output appears only after a silence endpoint (~1.3s of quiet)
- **Concurrency:** serialized — `WhisperModel.transcribe()` is not thread-safe, so a semaphore limits the model to one transcription at a time across all connections. Audio buffering and WebSocket I/O remain concurrent; only the transcription step queues.

#### Making Whisper fully concurrent

For production multi-user deployments there are two options:

1. **Multiple model instances** — instantiate one `WhisperModel` per worker process (via `uvicorn --workers N`). Each process owns its model and has no contention. Cost: N × VRAM.

2. **Dedicated inference server** — run [Triton Inference Server](https://github.com/triton-inference-server/server) or [whisper.cpp server](https://github.com/ggerganov/whisper.cpp) as a separate service. These handle dynamic batching internally, serving many concurrent requests from a single model instance efficiently. The transcriber session becomes a thin HTTP/gRPC client.
