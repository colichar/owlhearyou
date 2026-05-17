from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from src.transcriber import NemotronService

app = FastAPI()
# Initialize the engine once when the server starts
transcriber = NemotronService()

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    print("🚀 Client connected via WebSocket")
    
    try:
        while True:
            # Receive binary audio from your AudioRecorder
            audio_bytes = await websocket.receive_bytes()
            
            # Send to the independent engine
            text, is_final = await transcriber.transcribe_stream(audio_bytes)
            if text:
                await websocket.send_text(text + ("\n" if is_final else ""))
                
    except WebSocketDisconnect:
        print("🔌 Client disconnected")