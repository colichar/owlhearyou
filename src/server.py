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
            text = await transcriber.transcribe_stream(audio_bytes)
            
            # Send the result back to the client
            if text:
                await websocket.send_text(text)
                
    except WebSocketDisconnect:
        print("🔌 Client disconnected")