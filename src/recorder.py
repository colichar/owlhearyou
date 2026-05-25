import asyncio
import os
import sounddevice as sd
import logging
import numpy as np
from typing import AsyncIterator

logger = logging.getLogger(__name__)

class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.stream = None
        self.audio_queue = asyncio.Queue()
        self.loop = asyncio.get_event_loop()

    def _audio_callback(self, indata, frames, time, status):
        """This runs in a separate C-thread managed by sounddevice."""
        if status:
            logger.warning(f"Stream status: {status}")
        
        if self.is_recording:
            # Use call_soon_threadsafe to talk to the asyncio Queue
            # Send raw float32 bytes for transcription service to process
            self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, indata.copy())

    async def start_recording(self):
        if self.is_recording:
            return
        
        self.is_recording = True
        # Clear queue to ensure no old data is present
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()

        # None lets PortAudio pick the system default; AUDIO_DEVICE overrides for
        # systems where autodetection fails (e.g. specific PipeWire/ALSA setups).
        device = os.environ.get("AUDIO_DEVICE") or None

        try:
            self.stream = sd.InputStream(
                device=device,
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self._audio_callback,
                dtype='float32',
                blocksize=4000
            )
            self.stream.start()
        except sd.PortAudioError as e:
            # Reset flag so the caller can retry after fixing the device.
            self.is_recording = False
            # Query here rather than at startup so the list reflects the actual
            # state at the moment the error occurs (devices can be hotplugged).
            inputs = [
                f"  [{i}] {d['name']}"
                for i, d in enumerate(sd.query_devices())
                if d['max_input_channels'] > 0
            ]
            hint = "\n".join(inputs) if inputs else "  (none detected)"
            raise RuntimeError(
                f"Could not open audio input device: {e}\n\n"
                f"Available input devices:\n{hint}\n\n"
                f"Plug in a microphone, or set AUDIO_DEVICE=<index> to select one."
            ) from e
        logger.info("Recording started.")

    async def stop_recording(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        logger.info("Recording stopped.")

    async def stream_audio(self) -> AsyncIterator[bytes]:
        """
        An async generator that yields audio chunks.
        Perfect for FastAPI WebSockets or local processing.
        """
        while self.is_recording or not self.audio_queue.empty():
            try:
                # Wait for a chunk with a timeout so we can check is_recording
                chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.5)
                yield chunk.tobytes()
            except asyncio.TimeoutError:
                continue

# --- Manual test to verify the recording and streaming logic works as expected ---
async def run_local_transcription():
    recorder = AudioRecorder()
    await recorder.start_recording()
    
    all_chunks = []
    chunk_count = 0
    print("Speak now... (Press Ctrl+C to stop and save)")
    
    try:
        async for audio_chunk in recorder.stream_audio():
            all_chunks.append(audio_chunk)
            chunk_count += 1
            
            # Verification: Check volume (amplitude) of the current chunk
            audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
            amplitude = np.max(np.abs(audio_data))
            
            # Visual feedback: Chunk count + a simple volume bar
            bar = "!" * int(amplitude * 50)
            print(f"Chunk: {chunk_count:04} | Vol: {amplitude:.4f} {bar:<50}", end='\r')
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await recorder.stop_recording()
        
        # Save the result to verify the audio quality
        if all_chunks:
            full_audio = b"".join(all_chunks)
            # Use your existing conversion logic or a quick save:
            save_path = "verification_output.wav"
            write_wav(save_path, full_audio, recorder.sample_rate)
            print(f"\n✅ Saved {len(all_chunks)} chunks to {save_path}")

def write_wav(path, raw_data, samplerate):
    import wave
    audio_array = np.frombuffer(raw_data, dtype=np.float32)
    # Convert to int16 for standard WAV players
    int16_audio = (audio_array * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(int16_audio.tobytes())

if __name__ == "__main__":
    asyncio.run(run_local_transcription())