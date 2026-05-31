import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.client import OwlClient, transcribe


class MockWebSocket:
    def __init__(self, messages=()):
        self._messages = list(messages)

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for msg in self._messages:
            yield msg

    async def send(self, data):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


async def _blocking_stream():
    """Blocks forever so the receive side finishes first."""
    await asyncio.sleep(float("inf"))
    yield


def make_mock_recorder(stream=None):
    recorder = MagicMock()
    recorder.start_recording = AsyncMock()
    recorder.stop_recording = AsyncMock()
    recorder.stream_audio.return_value = stream if stream is not None else _blocking_stream()
    return recorder


# --- OwlClient.stt_stream() ---

async def test_stream_yields_partial_tuple():
    ws = MockWebSocket(["hello"])
    recorder = make_mock_recorder()

    with patch("src.client.websockets.connect", return_value=ws), \
         patch("src.client.AudioRecorder", return_value=recorder):
        results = []
        async for text, is_final in OwlClient("ws://localhost").stt_stream():
            results.append((text, is_final))

    assert results == [("hello", False)]


async def test_stream_yields_final_tuple():
    ws = MockWebSocket(["hello\n"])
    recorder = make_mock_recorder()

    with patch("src.client.websockets.connect", return_value=ws), \
         patch("src.client.AudioRecorder", return_value=recorder):
        results = []
        async for text, is_final in OwlClient("ws://localhost").stt_stream():
            results.append((text, is_final))

    assert results == [("hello", True)]


async def test_stream_stop_recording_called_in_finally():
    ws = MockWebSocket([])
    recorder = make_mock_recorder()

    with patch("src.client.websockets.connect", return_value=ws), \
         patch("src.client.AudioRecorder", return_value=recorder):
        async for _ in OwlClient("ws://localhost").stt_stream():
            pass

    recorder.stop_recording.assert_called_once()


# --- transcribe() print rendering ---

async def test_final_message_is_printed_with_rstrip():
    ws = MockWebSocket(["hello\n"])
    recorder = make_mock_recorder()

    with patch("src.client.websockets.connect", return_value=ws), \
         patch("src.client.AudioRecorder", return_value=recorder), \
         patch("builtins.print") as mock_print:
        await transcribe("ws://localhost")

    mock_print.assert_any_call("\rhello")


async def test_partial_message_is_printed_inline():
    ws = MockWebSocket(["hello"])
    recorder = make_mock_recorder()

    with patch("src.client.websockets.connect", return_value=ws), \
         patch("src.client.AudioRecorder", return_value=recorder), \
         patch("builtins.print") as mock_print:
        await transcribe("ws://localhost")

    mock_print.assert_any_call("\rhello", end="", flush=True)


async def test_stop_recording_always_called():
    ws = MockWebSocket([])
    recorder = make_mock_recorder()

    with patch("src.client.websockets.connect", return_value=ws), \
         patch("src.client.AudioRecorder", return_value=recorder):
        await transcribe("ws://localhost")

    recorder.stop_recording.assert_called_once()
