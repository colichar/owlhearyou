import asyncio
import websockets
from typing import AsyncIterator
from src.recorder import AudioRecorder


async def _send_audio(ws, recorder: AudioRecorder) -> None:
    async for chunk in recorder.stream_audio():
        await ws.send(chunk)


class OwlClient:
    def __init__(self, uri: str):
        self._uri = uri

    async def stream(self) -> AsyncIterator[tuple[str, bool]]:
        recorder = AudioRecorder()
        async with websockets.connect(self._uri) as ws:
            await recorder.start_recording()
            send_task = asyncio.create_task(_send_audio(ws, recorder))
            try:
                async for message in ws:
                    is_final = message.endswith("\n")
                    yield message.rstrip(), is_final
            finally:
                send_task.cancel()
                await recorder.stop_recording()


async def stream_to_server(uri: str) -> None:
    client = OwlClient(uri)
    try:
        async for text, is_final in client.stream():
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


if __name__ == "__main__":
    import argparse
    import sounddevice as sd
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="ws://localhost:8000/ws/transcribe")
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

    asyncio.run(stream_to_server(args.uri))
