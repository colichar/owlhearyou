#!/usr/bin/env python3
"""
Quick manual test for audio recording capability
Run this script to test the AudioRecorder manually
"""
import asyncio

from src.recorder import AudioRecorder

async def manual_audio_test():
    """Manual test of audio recording"""
    try:
        print("🎤 Manual Audio Recording Test")
        print("=============================")

        # Create audio recorder
        print("Creating audio recorder...")
        audio_recorder = AudioRecorder()

        # Check if audio is available
        if hasattr(audio_recorder, 'audio_available'):
            if audio_recorder.audio_available:
                print("✅ Audio recording is available on this system")
            else:
                print("❌ Audio recording is NOT available on this system")
                print("   (This might be due to missing audio devices or permissions)")
                return
        else:
            print("ℹ️  Audio availability check not implemented")

        print("\n🎤 Starting audio recording...")
        print("🎤 Speak into your microphone now...")
        print("🎤 Press Enter to stop recording...")

        # Start recording
        await audio_recorder.start_recording()

        # Collect chunks from the async generator in the background
        collected = []
        async def collect():
            async for chunk in audio_recorder.stream_audio():
                collected.append(chunk)

        collect_task = asyncio.create_task(collect())

        # Wait for Enter without blocking the event loop
        await asyncio.to_thread(input, "🎤 Recording... Press Enter to stop: ")

        print("🛑 Stopping recording...")
        await audio_recorder.stop_recording()
        collect_task.cancel()

        audio_data = b"".join(collected)

        if audio_data:
            print(f"🎉 Successfully recorded {len(audio_data)} bytes of audio!")
            print("💾 Audio data is ready to be sent to transcription service")
        else:
            print("⚠️ No audio data was recorded")
            print("   This could be due to:")
            print("   - No microphone connected")
            print("   - Microphone permissions not granted")
            print("   - Audio device configuration issues")

    except Exception as e:
        print(f"❌ Error during audio recording: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Manual Audio Recording Test")
    print("===========================")
    print("This script will test the AudioRecorder functionality")
    print("Make sure you have a microphone connected and permissions granted\n")

    asyncio.run(manual_audio_test())

    print("\nTest completed!")
    print("You can now test the full integration by running:")
    print("python main.py")
    print("Then type '\\audio' to start recording")