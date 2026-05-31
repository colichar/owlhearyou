import asyncio
import numpy as np
import websockets
from typing import AsyncIterator
from src.recorder import AudioRecorder

_TTS_SAMPLE_RATE = 24000


async def _send_audio(ws, recorder: AudioRecorder) -> None:
    async for chunk in recorder.stream_audio():
        await ws.send(chunk)


class OwlClient:
    def __init__(self, base_uri: str):
        self._base = base_uri.rstrip("/")

    async def stt_stream(self) -> AsyncIterator[tuple[str, bool]]:
        recorder = AudioRecorder()
        async with websockets.connect(f"{self._base}/ws/transcribe") as ws:
            await recorder.start_recording()
            send_task = asyncio.create_task(_send_audio(ws, recorder))
            try:
                async for message in ws:
                    is_final = message.endswith("\n")
                    yield message.rstrip(), is_final
            finally:
                send_task.cancel()
                await recorder.stop_recording()

    async def tts_stream(self, text: str, voice: str | None = None) -> AsyncIterator[bytes]:
        uri = f"{self._base}/ws/synthesize"
        if voice:
            uri += f"?voice={voice}"
        async with websockets.connect(uri) as ws:
            await ws.send(text)
            while True:
                chunk = await ws.recv()
                if not chunk:
                    break
                yield chunk


async def transcribe(base_uri: str) -> None:
    client = OwlClient(base_uri)
    try:
        async for text, is_final in client.stt_stream():
            if is_final:
                print(f"\r{text}")
            else:
                print(f"\r{text}", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopping stream...")
    except RuntimeError as e:
        # RuntimeError is raised by AudioRecorder when PortAudio fails to open
        # the input device — the message already contains the device list hint.
        print(f"\nAudio error: {e}")
    except Exception as e:
        # Catch-all for WebSocket and network failures.
        print(f"Connection error: {e}")


async def speak(text: str, base_uri: str, voice: str | None = None) -> None:
    import sounddevice as sd
    client = OwlClient(base_uri)
    stream = sd.OutputStream(samplerate=_TTS_SAMPLE_RATE, channels=1, dtype="float32")
    stream.start()
    try:
        async for chunk in client.tts_stream(text, voice=voice):
            stream.write(np.frombuffer(chunk, dtype="float32"))
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    import argparse
    import sounddevice as sd

    parser = argparse.ArgumentParser(description="Owl client — stream STT or synthesize TTS via owlhearyou server")
    parser.add_argument("--uri", default="ws://localhost:8000",
                        help="Server base URI (default: ws://localhost:8000)")
    parser.add_argument("--mode", choices=["stt", "tts"], default="stt",
                        help="stt: stream mic transcription; tts: synthesize text to speaker")
    parser.add_argument("--text", help="Text to synthesize (required in tts mode)")
    parser.add_argument("--voice", default=None, help="TTS voice name, e.g. af_heart")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio input devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        inputs = [(i, d["name"]) for i, d in enumerate(sd.query_devices())
                  if d["max_input_channels"] > 0]
        if inputs:
            for i, name in inputs:
                print(f"  [{i}] {name}")
        else:
            print("No input devices found. Check that a microphone is connected.")
        raise SystemExit(0)

    if args.mode == "tts":
        if not args.text:
            parser.error("--text is required in tts mode")
        asyncio.run(speak(args.text, args.uri, voice=args.voice))
    else:
        asyncio.run(transcribe(args.uri))
