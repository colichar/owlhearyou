#!/usr/bin/env python3
"""
Real-time transcription test using microphone locally (no WebSocket).
Selects backend via STT_BACKEND env var (nemotron | whisper). Press Ctrl+C to stop.
"""
import asyncio
import os

from src.recorder import AudioRecorder
from src.transcriber import NemotronService, WhisperService

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "nemotron")


async def main():
    backend = os.environ.get("STT_BACKEND", "nemotron")

    if backend == "whisper":
        model_size = os.environ.get("WHISPER_MODEL", "base")
        print(f"Loading Whisper model ({model_size})...")
        transcriber = WhisperService(model_size=model_size)
        print("Listening... results appear after each pause (Ctrl+C to stop)\n")
    else:
        print("Loading Nemotron model...")
        transcriber = NemotronService(model_dir=MODEL_DIR)
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
