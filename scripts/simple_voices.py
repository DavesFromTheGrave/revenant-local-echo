#!/usr/bin/env python3
"""Generate voice samples using Kokoro's minimal imports."""

import subprocess
import sys

# Just install spacy model quietly
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"], 
               capture_output=True)

try:
    from kokoro import build
    import sounddevice as sd
    import numpy as np
except ImportError as e:
    print(f"Missing: {e}")
    sys.exit(1)

VOICES = ['af_bella', 'af_sarah', 'af_nicole', 'am_adam', 'am_michael']
TEXT = "Hello, I'm your voice assistant. How can I help you today?"

print("Loading Kokoro model...")
try:
    model = build("kokoro-v0_19.pth", "cpu")
    print("✓ Model loaded\n")
except Exception as e:
    print(f"✗ Failed to load: {e}")
    sys.exit(1)

for voice in VOICES:
    try:
        print(f"Playing: {voice}")
        samples, sr = model.synthesize(TEXT, voice=voice)
        
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        
        max_val = np.abs(samples).max()
        if max_val > 1.0:
            samples = samples / max_val
        
        sd.play(samples, sr)
        sd.wait()
        print(f"  ✓ Done\n")
        
    except Exception as e:
        print(f"  ✗ Error: {e}\n")

print("Finished!")
