#!/usr/bin/env python3
"""
Kokoro voice sampler - test all available voices.
Uses the official KPipeline from the kokoro package.
"""
import torch
import sounddevice as sd
import numpy as np
from kokoro import KPipeline
import time

SAMPLE_TEXT = "Hello, I'm your voice assistant. How can I help you today?"

# Available Kokoro voices
FEMALE_VOICES = ["af_bella", "af_sarah", "af_nicole", "af_britain"]
MALE_VOICES = ["am_adam", "am_michael"]

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

def play_voice_sample(pipeline, voice_name):
    """Generate and play a voice sample."""
    print(f"\nVoice: {voice_name}")
    print("-" * 60)
    
    try:
        print("  Synthesizing...", end=" ", flush=True)

        # Generate audio using KPipeline
        generator = pipeline(SAMPLE_TEXT, voice=voice_name, speed=1.0)
        audio_chunks = []

        for gs, ps, audio in generator:
            audio_chunks.append(audio)

        if not audio_chunks:
            print("[FAIL] No audio generated")
            return

        # Concatenate all chunks
        full_audio = np.concatenate(audio_chunks)
        print("[OK]")
        
        # Normalize
        full_audio = normalize_audio(full_audio)
        
        # Play
        sample_rate = 24000
        duration = len(full_audio) / sample_rate
        print(f"  Playing ({duration:.2f}s)...", end=" ", flush=True)
        sd.play(full_audio, sample_rate)
        sd.wait()
        print("[OK]")
        
    except Exception as e:
        print(f"\n  [ERROR] {e}")
        import traceback
        traceback.print_exc()

def main():
    print("\n" + "="*60)
    print("KOKORO VOICE SAMPLER")
    print("="*60)
    
    # Initialize pipeline once (auto-downloads model if needed)
    print("\nInitializing Kokoro pipeline...", end=" ", flush=True)
    try:
        pipeline = KPipeline(lang_code='a')  # 'a' = American English
        print("[OK]")
    except Exception as e:
        print(f"[FAIL]\nFailed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nSample text: \"{SAMPLE_TEXT}\"\n")
    
    print("FEMALE VOICES:")
    print("-" * 60)
    for voice_name in FEMALE_VOICES:
        play_voice_sample(pipeline, voice_name)
        time.sleep(0.5)  # Brief pause between voices
    
    print("\n\nMALE VOICES:")
    print("-" * 60)
    for voice_name in MALE_VOICES:
        play_voice_sample(pipeline, voice_name)
        time.sleep(0.5)  # Brief pause between voices
    
    print("\n" + "="*60)
    print("Sampling complete!")
    print("="*60)

if __name__ == "__main__":
    main()
