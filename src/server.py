from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from src.transcriber import NemotronService

app = FastAPI()
# Initialize the engine once when the server starts
transcriber = NemotronService()

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    print("Client connected via WebSocket")
    session = transcriber.create_session()

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            text, is_final = await session.transcribe(audio_bytes)
            if text:
                await websocket.send_text(text + ("\n" if is_final else ""))

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        await session.close()