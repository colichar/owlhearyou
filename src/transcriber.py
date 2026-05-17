import asyncio
import sherpa_onnx
import numpy as np
import os

class NemotronService:
    def __init__(self, model_dir="../models/nemotron"):
        encoder = os.path.join(model_dir, "encoder.int8.onnx")
        decoder = os.path.join(model_dir, "decoder.int8.onnx")
        joiner = os.path.join(model_dir, "joiner.int8.onnx")
        tokens = os.path.join(model_dir, "tokens.txt")

        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            provider="cuda",
            device=0,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=20,
        )

        self.stream = self.recognizer.create_stream()

    def _decode_chunk(self, samples: np.ndarray) -> tuple[str, bool]:
        self.stream.accept_waveform(16000, samples)
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        text = self.recognizer.get_result(self.stream)
        is_final = self.recognizer.is_endpoint(self.stream)
        if is_final:
            self.recognizer.reset(self.stream)
        return text.strip(), is_final

    async def transcribe_stream(self, audio_bytes: bytes) -> tuple[str, bool]:
        samples = np.frombuffer(audio_bytes, dtype=np.float32)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._decode_chunk, samples)