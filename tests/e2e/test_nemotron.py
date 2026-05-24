import os
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("STT_BACKEND", "whisper") != "nemotron",
    reason="STT_BACKEND is not 'nemotron'",
)


async def test_transcribes_speech(transcribe, hello_en_chunks):
    results = await transcribe(hello_en_chunks)

    assert results, "No transcription received from server"
    assert any(is_final for _, is_final in results), "Never received a final result"

    text = " ".join(t for t, _ in results).strip().lower()
    assert text, "Transcription text was empty"
