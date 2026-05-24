import asyncio

import numpy as np
import websockets


_EXPECTED_KEYWORDS = {
    "test_case_1": ("nightfall", "yellow", "lamps"),
    "test_case_2": ("affected", "thoughts", "hope"),
}


def _assert_transcript(results, keywords):
    assert results, "No transcription received from server"
    # Use the final result only — partials are growing prefixes (Nemotron) or absent (Whisper)
    final = next((t for t, is_final in results if is_final), None)
    assert final is not None, "Never received a final result"
    text = final.strip().lower()
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


async def test_silence_produces_no_output(server_uri):
    messages = []
    silence = np.zeros(512, dtype=np.float32).tobytes()

    async with websockets.connect(server_uri) as ws:
        async def _collect():
            async for msg in ws:
                messages.append(msg)

        collect_task = asyncio.create_task(_collect())
        for _ in range(20):
            await ws.send(silence)
        collect_task.cancel()

    assert not messages, f"Expected no output for silence, got: {messages}"
