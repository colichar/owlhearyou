import asyncio
import sherpa_onnx
import numpy as np
import os
from src.base_transcriber import BaseTranscriber, TranscriptionSession

_MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))
_NEMOTRON_MODEL_DIR = os.path.join(_MODELS_DIR, "nemotron")
_WHISPER_MODEL_DIR = os.path.join(_MODELS_DIR, "whisper")


class WhisperSession(TranscriptionSession):
    _SILENCE_RMS = 0.01
    _SILENCE_CHUNKS = 40   # ~1.3s at 32ms/chunk
    _MAX_SECONDS = 30

    def __init__(self, model, semaphore: asyncio.Semaphore, sample_rate: int = 16000, language: str | None = None):
        self._model = model
        self._semaphore = semaphore
        self._sample_rate = sample_rate
        self._language = language
        self._buffer: list[np.ndarray] = []
        self._silent_chunks = 0

    def _should_flush(self, samples: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(samples ** 2)))
        self._silent_chunks = self._silent_chunks + 1 if rms < self._SILENCE_RMS else 0
        buffered_seconds = sum(len(s) for s in self._buffer) / self._sample_rate
        return self._silent_chunks >= self._SILENCE_CHUNKS or buffered_seconds >= self._MAX_SECONDS

    def _flush(self) -> tuple[str, bool]:
        audio = np.concatenate(self._buffer)
        self._buffer.clear()
        self._silent_chunks = 0
        segments, _ = self._model.transcribe(audio, vad_filter=True, language=self._language, task="transcribe")
        return " ".join(s.text for s in segments).strip(), True

    async def transcribe(self, audio_bytes: bytes) -> tuple[str, bool]:
        samples = np.frombuffer(audio_bytes, dtype=np.float32)
        self._buffer.append(samples)
        if not self._should_flush(samples):
            return "", False
        loop = asyncio.get_running_loop()
        async with self._semaphore:
            return await loop.run_in_executor(None, self._flush)

    async def close(self) -> None:
        self._buffer.clear()


class WhisperService(BaseTranscriber):
    def __init__(self, model_size: str = "base", device: str = "auto", language: str | None = None):
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model_size, device=device, download_root=_WHISPER_MODEL_DIR)
        self._semaphore = asyncio.Semaphore(1)
        self._language = language

    def create_session(self) -> WhisperSession:
        return WhisperSession(self._model, self._semaphore, language=self._language)


class NemotronSession(TranscriptionSession):
    def __init__(self, recognizer: sherpa_onnx.OnlineRecognizer):
        self._recognizer = recognizer
        self._stream = recognizer.create_stream()

    def _decode_chunk(self, samples: np.ndarray) -> tuple[str, bool]:
        self._stream.accept_waveform(16000, samples)
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        text = self._recognizer.get_result(self._stream)
        is_final = self._recognizer.is_endpoint(self._stream)
        if is_final:
            self._recognizer.reset(self._stream)
        return text.strip(), is_final

    async def transcribe(self, audio_bytes: bytes) -> tuple[str, bool]:
        samples = np.frombuffer(audio_bytes, dtype=np.float32)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._decode_chunk, samples)

    async def close(self) -> None:
        pass


# Pinned to the specific int8 quantized variant this project was built and tested against.
# Changing the model requires re-validating the recognizer config below.
_NEMOTRON_REPO = "csukuangfj/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14"


class NemotronService(BaseTranscriber):
    def __init__(self, model_dir=_NEMOTRON_MODEL_DIR, provider: str = "cuda"):
        if not os.path.exists(os.path.join(model_dir, "encoder.int8.onnx")):
            from huggingface_hub import snapshot_download
            print(f"Downloading Nemotron model to {model_dir}...")
            snapshot_download(repo_id=_NEMOTRON_REPO, local_dir=model_dir)

        encoder = os.path.join(model_dir, "encoder.int8.onnx")
        decoder = os.path.join(model_dir, "decoder.int8.onnx")
        joiner = os.path.join(model_dir, "joiner.int8.onnx")
        tokens = os.path.join(model_dir, "tokens.txt")

        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            provider=provider,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=20,
        )

    def create_session(self) -> NemotronSession:
        return NemotronSession(self._recognizer)
