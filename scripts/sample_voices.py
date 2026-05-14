#!/usr/bin/env python3
"""Generate voice samples for each Kokoro voice."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from kokoro import build
    import sounddevice as sd
    import numpy as np
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure dependencies are installed")
    sys.exit(1)

# Voices to test
VOICES = ['af_bella', 'af_sarah', 'af_nicole', 'am_adam', 'am_michael']

# Sample text
SAMPLE_TEXT = "Hello, I'm your voice assistant. How can I help you today?"

def main():
    print("Generating voice samples...\n")
    
    try:
        model = build("kokoro-v0_19.pth", "cpu")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    
    for voice in VOICES:
        try:
            print(f"Generating sample for {voice}...")
            samples, sr = model.synthesize(SAMPLE_TEXT, voice=voice)
            
            # Ensure audio is float32 in [-1, 1] range
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32)
            
            # Normalize if needed
            max_val = np.abs(samples).max()
            if max_val > 1.0:
                samples = samples / max_val
            
            # Play audio
            sd.play(samples, sr)
            sd.wait()
            print(f"  ✓ Played {voice}\n")
            
        except Exception as e:
            print(f"  ✗ Error with {voice}: {e}\n")
    
    print("Done!")

if __name__ == "__main__":
    main()
