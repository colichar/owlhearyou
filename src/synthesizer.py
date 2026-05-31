import os
import urllib.request
from abc import ABC, abstractmethod
from typing import AsyncIterator

_MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))
_KOKORO_MODEL_DIR = os.path.join(_MODELS_DIR, "kokoro")

_KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
_KOKORO_MODEL_FILE = "kokoro-v1.0.onnx"
_KOKORO_VOICES_FILE = "voices-v1.0.bin"


class BaseSynthesizer(ABC):
    @abstractmethod
    def synthesize_stream(self, text: str, voice: str) -> AsyncIterator[bytes]:
        ...


class KokoroService(BaseSynthesizer):
    def __init__(self, model_dir: str = _KOKORO_MODEL_DIR, lang: str = "en-us"):
        from kokoro_onnx import Kokoro

        model_path = os.path.join(model_dir, _KOKORO_MODEL_FILE)
        voices_path = os.path.join(model_dir, _KOKORO_VOICES_FILE)

        if not os.path.exists(model_path) or not os.path.exists(voices_path):
            print(f"Downloading Kokoro to {model_dir}...")
            os.makedirs(model_dir, exist_ok=True)
            urllib.request.urlretrieve(f"{_KOKORO_RELEASE}/{_KOKORO_MODEL_FILE}", model_path)
            urllib.request.urlretrieve(f"{_KOKORO_RELEASE}/{_KOKORO_VOICES_FILE}", voices_path)

        self._kokoro = Kokoro(model_path, voices_path)
        self._lang = lang

    async def synthesize_stream(self, text: str, voice: str) -> AsyncIterator[bytes]:
        async for samples, _ in self._kokoro.create_stream(text, voice=voice, speed=1.0, lang=self._lang):
            yield samples.astype("float32").tobytes()
