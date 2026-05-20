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
    except Exception as e:
        print(f"Connection error: {e}")


if __name__ == "__main__":
    SERVER_URI = "ws://localhost:8000/ws/transcribe"
    asyncio.run(stream_to_server(SERVER_URI))
