#!/usr/bin/env python3
"""Test af_nicole at speeds 1.0, 1.15, and 1.25."""
import sounddevice as sd
import numpy as np
from kokoro import KPipeline
import time

SAMPLE_TEXT = "Hello, I'm your voice assistant. How can I help you today?"

def normalize_audio(audio, target_db=-20.0):
    """Normalize audio to target loudness."""
    if len(audio) == 0:
        return audio
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        db = 20 * np.log10(rms)
        gain = 10 ** ((target_db - db) / 20)
        audio = audio * gain
    return np.clip(audio, -1.0, 1.0)

def test_speed(pipeline, speed):
    """Test af_nicole at a specific speed."""
    print(f"\nSpeed: {speed}")
    print("-" * 40)

    try:
        print("  Synthesizing...", end=" ", flush=True)
        generator = pipeline(SAMPLE_TEXT, voice="af_nicole", speed=speed)
        audio_chunks = []

        for gs, ps, audio in generator:
            audio_chunks.append(audio)

        if not audio_chunks:
            print("[FAIL] No audio generated")
            return

        full_audio = np.concatenate(audio_chunks)
        print("[OK]")

        full_audio = normalize_audio(full_audio)
        sample_rate = 24000
        duration = len(full_audio) / sample_rate

        print(f"  Playing ({duration:.2f}s)...", end=" ", flush=True)
        sd.play(full_audio, sample_rate)
        sd.wait()
        print("[OK]")

    except Exception as e:
        print(f"[ERROR] {e}")

def main():
    print("\n" + "="*60)
    print("NICOLE SPEED COMPARISON")
    print("="*60)

    print("\nInitializing Kokoro pipeline...", end=" ", flush=True)
    try:
        pipeline = KPipeline(lang_code='a')
        print("[OK]")
    except Exception as e:
        print(f"[FAIL]\nFailed to initialize: {e}")
        return

    print(f"\nSample text: \"{SAMPLE_TEXT}\"\n")

    speeds = [1.0, 1.15, 1.25]
    for speed in speeds:
        test_speed(pipeline, speed)
        time.sleep(0.5)

    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)

if __name__ == "__main__":
    main()
