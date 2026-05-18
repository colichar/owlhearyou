import numpy as np
from unittest.mock import MagicMock
from src.transcriber import NemotronService, NemotronSession


def make_session(get_result="hello", is_endpoint=False, is_ready=False):
    recognizer = MagicMock()
    recognizer.is_ready.return_value = is_ready
    recognizer.get_result.return_value = get_result
    recognizer.is_endpoint.return_value = is_endpoint
    return NemotronSession(recognizer)


SAMPLES = np.zeros(100, dtype=np.float32)


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
