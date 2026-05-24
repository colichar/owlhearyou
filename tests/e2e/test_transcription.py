async def test_transcribes_speech(transcribe, hello_en_chunks, hello_en_transcript):
    results = await transcribe(hello_en_chunks)

    assert results, "No transcription received from server"
    assert any(is_final for _, is_final in results), "Never received a final result"

    text = " ".join(t for t, _ in results).strip().lower()
    for word in ("nightfall", "yellow", "lamps"):
        assert word in text, f"Expected '{word}' in transcript, got: {text!r}"
