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
            device=0
        )

        self.stream = self.recognizer.create_stream()
        self.last_text = ""

    async def transcribe_stream(self, audio_bytes: bytes) -> str:
        """Processes audio and returns only NEW words."""
        samples = np.frombuffer(audio_bytes, dtype=np.float32)
        
        # Feed audio to the stateful stream
        self.stream.accept_waveform(16000, samples)
        
        # Decode the updated state
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        
        # Get full current transcript and return only what's new
        full_text = self.recognizer.get_result(self.stream) # .text
        new_text = full_text[len(self.last_text):].strip()
        self.last_text = full_text
        
        return new_text