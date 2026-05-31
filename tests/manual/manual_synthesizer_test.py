#!/usr/bin/env python3
"""
Real-time TTS test using KokoroService locally (no WebSocket).
Prompts for text, synthesizes audio, plays it via the default output device.
Press Ctrl+C or send an empty line to stop.
"""
import asyncio
import os

import numpy as np
import sounddevice as sd

from src.synthesizer import KokoroService

SAMPLE_RATE = 24000


async def main():
    voice = os.environ.get("KOKORO_VOICE", "af_heart")
    print(f"Loading Kokoro model (voice: {voice})...")
    synth = KokoroService()
    print("Type text and press Enter to synthesize. Empty line or Ctrl+C to quit.\n")

    while True:
        try:
            text = await asyncio.to_thread(input, "> ")
        except (EOFError, KeyboardInterrupt):
            print("\nStopping...")
            return

        text = text.strip()
        if not text:
            print("Stopping...")
            return

        stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        stream.start()
        try:
            async for chunk in synth.synthesize_stream(text, voice=voice):
                stream.write(np.frombuffer(chunk, dtype="float32"))
        except Exception as e:
            print(f"Error: {e}")
        finally:
            stream.stop()
            stream.close()


if __name__ == "__main__":
    asyncio.run(main())
