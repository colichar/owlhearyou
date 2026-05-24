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
    with wave.open(path) as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sample_width]
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if sample_width == 1:
        samples = (samples - 128) / 128.0  # uint8 → [-1, 1]
    else:
        samples /= float(np.iinfo(dtype).max)

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    if sr != target_sr:
        new_len = int(len(samples) * target_sr / sr)
        samples = np.interp(
            np.linspace(0, len(samples) - 1, new_len),
            np.arange(len(samples)),
            samples,
        ).astype(np.float32)

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


_SILENCE_CHUNK = np.zeros(512, dtype=np.float32).tobytes()


@pytest.fixture(scope="session")
def transcribe(server_uri):
    async def _transcribe(chunks: list[bytes], silence_padding: int = 80) -> list[tuple[str, bool]]:
        results: list[tuple[str, bool]] = []

        async with websockets.connect(server_uri) as ws:
            async def _recv():
                async for msg in ws:
                    is_final = msg.endswith("\n")
                    results.append((msg.rstrip(), is_final))
                    if is_final:
                        return

            recv_task = asyncio.create_task(_recv())
            for chunk in chunks:
                await ws.send(chunk)
            for _ in range(silence_padding):
                await ws.send(_SILENCE_CHUNK)
            await asyncio.wait_for(recv_task, timeout=120)

        return results

    return _transcribe
