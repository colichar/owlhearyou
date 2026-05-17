#!/usr/bin/env python3
"""
Real-time transcription test using microphone + Nemotron locally (no WebSocket).
Press Ctrl+C to stop.
"""
import asyncio
import os

from src.recorder import AudioRecorder
from src.transcriber import NemotronService

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "nemotron")

async def main():
    print("Loading Nemotron model...")
    transcriber = NemotronService(model_dir=MODEL_DIR)
    recorder = AudioRecorder(sample_rate=16000, channels=1)

    await recorder.start_recording()
    print("Listening... (Ctrl+C to stop)\n")

    try:
        async for chunk in recorder.stream_audio():
            text, is_final = await transcriber.transcribe_stream(chunk)
            if text:
                if is_final:
                    print(f"\r{text}")
                else:
                    print(f"\r{text}", end="", flush=True)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        await recorder.stop_recording()

if __name__ == "__main__":
    asyncio.run(main())
