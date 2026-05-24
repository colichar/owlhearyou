#!/usr/bin/env python3
"""
Real-time transcription test using microphone locally (no WebSocket).
Selects backend via STT_BACKEND env var (nemotron | whisper). Press Ctrl+C to stop.
"""
import asyncio
import os

from src.recorder import AudioRecorder
from src.transcriber import NemotronService, WhisperService


async def main():
    backend = os.environ.get("STT_BACKEND", "nemotron")
    device = os.environ.get("STT_DEVICE", "cuda")
    onnx_provider = "cpu" if device == "cpu" else "cuda"

    if backend == "whisper":
        model_size = os.environ.get("WHISPER_MODEL", "base")
        language = os.environ.get("WHISPER_LANGUAGE") or None
        print(f"Loading Whisper model ({model_size}, device={device})...")
        transcriber = WhisperService(model_size=model_size, language=language, device=device)
        print("Listening... results appear after each pause (Ctrl+C to stop)\n")
    else:
        print(f"Loading Nemotron model (provider={onnx_provider})...")
        transcriber = NemotronService(provider=onnx_provider)
        print("Listening... (Ctrl+C to stop)\n")

    recorder = AudioRecorder(sample_rate=16000, channels=1)
    session = transcriber.create_session()

    await recorder.start_recording()

    try:
        async for chunk in recorder.stream_audio():
            text, is_final = await session.transcribe(chunk)
            if text:
                if is_final:
                    print(f"\r{text}")
                else:
                    print(f"\r{text}", end="", flush=True)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        await recorder.stop_recording()
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
