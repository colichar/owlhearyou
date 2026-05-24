import asyncio


_EXPECTED_KEYWORDS = {
    "test_case_1": ("nightfall", "yellow", "lamps"),
    "test_case_2": ("apprehension", "thoughts", "hope"),
}


def _assert_transcript(results, keywords):
    assert results, "No transcription received from server"
    assert any(is_final for _, is_final in results), "Never received a final result"
    text = " ".join(t for t, _ in results).strip().lower()
    for word in keywords:
        assert word in text, f"Expected '{word}' in transcript, got: {text!r}"


async def test_transcribes_speech(transcribe, test_cases):
    for (chunks, _), keywords in zip(test_cases, _EXPECTED_KEYWORDS.values()):
        results = await transcribe(chunks)
        _assert_transcript(results, keywords)


async def test_concurrent_sessions(transcribe, test_cases):
    (chunks_1, _), (chunks_2, _) = test_cases
    keywords_1, keywords_2 = _EXPECTED_KEYWORDS.values()

    results_a, results_b = await asyncio.gather(
        transcribe(chunks_1),
        transcribe(chunks_2),
    )

    _assert_transcript(results_a, keywords_1)
    _assert_transcript(results_b, keywords_2)
