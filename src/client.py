import asyncio
import websockets
from recorder import AudioRecorder

async def stream_to_server(uri: str):
    recorder = AudioRecorder()
    
    try:
        # 1. Connect to the FastAPI server
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            await recorder.start_recording()
            
            # 2. Start two concurrent tasks: one for sending, one for receiving
            # This allows us to see transcripts while we are still talking
            async def send_audio():
                async for chunk in recorder.stream_audio():
                    await websocket.send(chunk)
            
            async def receive_transcript():

                async for message in websocket:
                    print(f"Transcript: {message}", flush=True)

            done, pending = await asyncio.wait(
                [asyncio.create_task(send_audio()),
                 asyncio.create_task(receive_transcript())],
                 return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()


    except KeyboardInterrupt:
        print("\nStopping stream...")
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        await recorder.stop_recording()

if __name__ == "__main__":
    SERVER_URI = "ws://localhost:8000/ws/transcribe"
    asyncio.run(stream_to_server(SERVER_URI))