import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from src.transcriber import NemotronService, WhisperService
from src.synthesizer import KokoroService

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

synthesizer = KokoroService()
_default_voice = os.environ.get("KOKORO_VOICE", "af_heart")


@app.get("/health")
async def health():
    return {"status": "ok"}


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


@app.websocket("/ws/synthesize")
async def websocket_synthesize(websocket: WebSocket, voice: str = _default_voice):
    await websocket.accept()
    try:
        while True:
            text = await websocket.receive_text()
            async for chunk in synthesizer.synthesize_stream(text, voice=voice):
                await websocket.send_bytes(chunk)
            await websocket.send_bytes(b"")
    except WebSocketDisconnect:
        pass
