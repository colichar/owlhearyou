import asyncio
import numpy as np
import pytest
from unittest.mock import patch
from src.recorder import AudioRecorder


def make_recorder():
    return AudioRecorder(sample_rate=16000, channels=1)


# --- stream_audio() ---

async def test_stream_audio_yields_chunk_as_bytes():
    recorder = make_recorder()
    recorder.is_recording = False
    chunk = np.ones(4000, dtype=np.float32)
    recorder.audio_queue.put_nowait(chunk)

    results = []
    async for audio in recorder.stream_audio():
        results.append(audio)

    assert results == [chunk.tobytes()]


async def test_stream_audio_preserves_chunk_order():
    recorder = make_recorder()
    recorder.is_recording = False
    chunks = [np.full(100, float(i), dtype=np.float32) for i in range(3)]
    for c in chunks:
        recorder.audio_queue.put_nowait(c)

    results = []
    async for audio in recorder.stream_audio():
        results.append(audio)

    assert results == [c.tobytes() for c in chunks]


async def test_stream_audio_empty_queue_exits_immediately():
    recorder = make_recorder()
    recorder.is_recording = False

    results = []
    async for audio in recorder.stream_audio():
        results.append(audio)

    assert results == []


# --- start_recording() ---

async def test_start_recording_clears_stale_queue():
    with patch('src.recorder.sd.InputStream'):
        recorder = make_recorder()
        recorder.audio_queue.put_nowait(np.zeros(100, dtype=np.float32))

        await recorder.start_recording()

        assert recorder.audio_queue.empty()


async def test_start_recording_sets_is_recording_true():
    with patch('src.recorder.sd.InputStream'):
        recorder = make_recorder()
        assert not recorder.is_recording

        await recorder.start_recording()

        assert recorder.is_recording


async def test_start_recording_is_idempotent():
    with patch('src.recorder.sd.InputStream') as mock_cls:
        recorder = make_recorder()
        await recorder.start_recording()
        await recorder.start_recording()

        # InputStream should only be constructed once
        assert mock_cls.call_count == 1


# --- _audio_callback() ---

async def test_audio_callback_enqueues_data_when_recording():
    recorder = make_recorder()
    recorder.is_recording = True

    fake_data = np.ones((4000, 1), dtype=np.float32)
    recorder._audio_callback(fake_data, 4000, None, None)
    await asyncio.sleep(0)  # let call_soon_threadsafe flush

    assert not recorder.audio_queue.empty()
    item = recorder.audio_queue.get_nowait()
    np.testing.assert_array_equal(item, fake_data)


async def test_audio_callback_ignores_data_when_not_recording():
    recorder = make_recorder()
    recorder.is_recording = False

    recorder._audio_callback(np.ones((100, 1), dtype=np.float32), 100, None, None)
    await asyncio.sleep(0)

    assert recorder.audio_queue.empty()
