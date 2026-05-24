import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from src.transcriber import NemotronService, WhisperService

app = FastAPI()

_backend = os.environ.get("STT_BACKEND", "nemotron")
_device = os.environ.get("STT_DEVICE", "cuda")
_onnx_provider = "cpu" if _device == "cpu" else "cuda"

if _backend == "whisper":
    transcriber = WhisperService(
        model_size=os.environ.get("WHISPER_MODEL", "base"),
        language=os.environ.get("WHISPER_LANGUAGE") or None,
        device=_device,
    )
else:
    transcriber = NemotronService(provider=_onnx_provider)

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