import asyncio
import os
import threading
import time
import wave
import numpy as np
import pytest
import uvicorn
import websockets

_HERE = os.path.dirname(__file__)

# Set defaults before src.server is imported (it reads env vars at module level).
# Tests override these via env vars; CPU is the CI-safe default.
os.environ.setdefault("STT_BACKEND", "whisper")
os.environ.setdefault("STT_DEVICE", "cpu")
os.environ.setdefault("WHISPER_MODEL", "base")


@pytest.fixture(scope="session")
def server_uri():
    from src.server import app

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 60
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("Server did not start within 60s (model load timeout?)")
        time.sleep(0.1)

    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"ws://127.0.0.1:{port}/ws/transcribe"

    server.should_exit = True
    thread.join(timeout=5)


def _wav_to_chunks(path: str, chunk_ms: int = 32, target_sr: int = 16000) -> list[bytes]:
    # Read all frames and metadata from the WAV file
    with wave.open(path) as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    # Interpret raw bytes as the correct integer type, then widen to float32
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sample_width]
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    # Normalize to [-1.0, 1.0]: uint8 is unsigned so centre at 128, others are signed
    if sample_width == 1:
        samples = (samples - 128) / 128.0
    else:
        samples /= float(np.iinfo(dtype).max)

    # Downmix to mono by averaging across channels
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # Resample to target rate using linear interpolation
    if sr != target_sr:
        new_len = int(len(samples) * target_sr / sr)
        samples = np.interp(
            np.linspace(0, len(samples) - 1, new_len),
            np.arange(len(samples)),
            samples,
        ).astype(np.float32)

    # Split into fixed-size chunks and serialize each as raw float32 bytes
    chunk_size = int(target_sr * chunk_ms / 1000)
    return [
        samples[i : i + chunk_size].astype(np.float32).tobytes()
        for i in range(0, len(samples), chunk_size)
    ]


@pytest.fixture(scope="session")
def hello_en_chunks():
    path = os.path.join(_HERE, "fixtures", "hello_en.wav")
    if not os.path.exists(path):
        pytest.skip(f"Fixture not found: {path} — record a short English phrase and save it there")
    return _wav_to_chunks(path)


@pytest.fixture(scope="session")
def hello_en_transcript():
    path = os.path.join(_HERE, "fixtures", "hello_en.txt")
    if not os.path.exists(path):
        pytest.skip(f"Transcript fixture not found: {path}")
    return open(path).read().strip().lower()


_SILENCE_CHUNK = np.zeros(512, dtype=np.float32).tobytes()


@pytest.fixture(scope="session")
def transcribe(server_uri):
    async def _transcribe(chunks: list[bytes], silence_padding: int = 80) -> list[tuple[str, bool]]:
        results: list[tuple[str, bool]] = []

        async with websockets.connect(server_uri) as ws:
            async def _recv():
                # Collect server messages until the first final result then stop
                async for msg in ws:
                    is_final = msg.endswith("\n")
                    results.append((msg.rstrip(), is_final))
                    if is_final:
                        return

            # Start receiving concurrently so partial results aren't dropped while sending
            recv_task = asyncio.create_task(_recv())
            # Stream the audio chunks to the server
            for chunk in chunks:
                await ws.send(chunk)
            # Pad with silence to push both backends past their endpoint detection threshold
            for _ in range(silence_padding):
                await ws.send(_SILENCE_CHUNK)
            # Wait for the final result; 120s budget covers slow CPU inference
            await asyncio.wait_for(recv_task, timeout=120)

        return results

    return _transcribe
