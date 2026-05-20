import asyncio
import numpy as np
from unittest.mock import MagicMock
from src.transcriber import NemotronService, NemotronSession, WhisperService, WhisperSession


def make_session(get_result="hello", is_endpoint=False, is_ready=False):
    recognizer = MagicMock()
    recognizer.is_ready.return_value = is_ready
    recognizer.get_result.return_value = get_result
    recognizer.is_endpoint.return_value = is_endpoint
    return NemotronSession(recognizer)


SAMPLES = np.zeros(100, dtype=np.float32)


SILENT = np.zeros(512, dtype=np.float32)
LOUD = np.ones(512, dtype=np.float32) * 0.5


def make_whisper_session():
    model = MagicMock()
    return WhisperSession(model, asyncio.Semaphore(1))


def make_segments(*texts):
    segments = []
    for t in texts:
        seg = MagicMock()
        seg.text = t
        segments.append(seg)
    return segments


# --- WhisperSession._should_flush ---

def test_whisper_should_flush_false_before_silence_threshold():
    session = make_whisper_session()
    for _ in range(WhisperSession._SILENCE_CHUNKS - 1):
        assert session._should_flush(SILENT) is False


def test_whisper_should_flush_true_after_silence_threshold():
    session = make_whisper_session()
    for _ in range(WhisperSession._SILENCE_CHUNKS - 1):
        session._should_flush(SILENT)
    assert session._should_flush(SILENT) is True


def test_whisper_should_flush_resets_on_loud_chunk():
    session = make_whisper_session()
    for _ in range(WhisperSession._SILENCE_CHUNKS - 1):
        session._should_flush(SILENT)
    session._should_flush(LOUD)
    assert session._should_flush(SILENT) is False


def test_whisper_should_flush_true_at_max_seconds():
    session = make_whisper_session()
    session._buffer = [np.zeros(int(WhisperSession._MAX_SECONDS * 16000), dtype=np.float32)]
    assert session._should_flush(LOUD) is True


# --- WhisperSession._flush ---

def test_whisper_flush_returns_joined_segments():
    session = make_whisper_session()
    session._buffer = [SILENT]
    session._model.transcribe.return_value = (make_segments("hello", "world"), None)
    text, is_final = session._flush()
    assert text == "hello world"
    assert is_final is True


def test_whisper_flush_clears_buffer():
    session = make_whisper_session()
    session._buffer = [SILENT, LOUD]
    session._model.transcribe.return_value = ([], None)
    session._flush()
    assert session._buffer == []


# --- WhisperSession.transcribe ---

async def test_whisper_transcribe_returns_empty_when_not_flushing():
    session = make_whisper_session()
    text, is_final = await session.transcribe(LOUD.tobytes())
    assert text == ""
    assert is_final is False


async def test_whisper_transcribe_returns_result_on_flush():
    session = make_whisper_session()
    session._buffer = [np.zeros(int(WhisperSession._MAX_SECONDS * 16000), dtype=np.float32)]
    session._model.transcribe.return_value = (make_segments("hello"), None)
    text, is_final = await session.transcribe(LOUD.tobytes())
    assert text == "hello"
    assert is_final is True


async def test_whisper_close_clears_buffer():
    session = make_whisper_session()
    session._buffer = [SILENT, LOUD]
    await session.close()
    assert session._buffer == []


# --- WhisperService.create_session ---

def test_whisper_create_session_returns_correct_session():
    service = WhisperService.__new__(WhisperService)
    service._model = MagicMock()
    service._semaphore = asyncio.Semaphore(1)
    service._language = None
    session = service.create_session()
    assert isinstance(session, WhisperSession)
    assert session._model is service._model
    assert session._semaphore is service._semaphore


# --- NemotronSession._decode_chunk ---

def test_decode_chunk_returns_text_and_is_final():
    session = make_session(get_result="hello world", is_endpoint=False)
    text, is_final = session._decode_chunk(SAMPLES)
    assert text == "hello world"
    assert is_final is False


def test_decode_chunk_strips_whitespace():
    session = make_session(get_result="  hello  ")
    text, _ = session._decode_chunk(SAMPLES)
    assert text == "hello"


def test_decode_chunk_resets_stream_when_endpoint():
    session = make_session(is_endpoint=True)
    _, is_final = session._decode_chunk(SAMPLES)
    assert is_final is True
    session._recognizer.reset.assert_called_once_with(session._stream)


def test_decode_chunk_no_reset_when_not_endpoint():
    session = make_session(is_endpoint=False)
    session._decode_chunk(SAMPLES)
    session._recognizer.reset.assert_not_called()


def test_decode_chunk_drains_decoder_while_ready():
    recognizer = MagicMock()
    recognizer.is_ready.side_effect = [True, True, False]
    recognizer.get_result.return_value = ""
    recognizer.is_endpoint.return_value = False
    session = NemotronSession(recognizer)
    session._decode_chunk(SAMPLES)
    assert recognizer.decode_stream.call_count == 2


# --- NemotronSession.transcribe ---

async def test_transcribe_returns_decoded_tuple():
    session = make_session(get_result="test", is_endpoint=True)
    result = await session.transcribe(SAMPLES.tobytes())
    assert result == ("test", True)


async def test_close_does_not_raise():
    session = make_session()
    await session.close()


# --- NemotronService.create_session ---

def test_create_session_returns_nemotron_session_with_correct_recognizer():
    service = NemotronService.__new__(NemotronService)
    mock_recognizer = MagicMock()
    service._recognizer = mock_recognizer
    session = service.create_session()
    assert isinstance(session, NemotronSession)
    assert session._recognizer is mock_recognizer
