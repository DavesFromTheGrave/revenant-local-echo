"""
Enumerate all Kokoro voices and generate sample audio files.
Run this once to hear all available voices and choose your favorite.

Usage:
    python enumerate_voices.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tts import TTS
from config_loader import VoiceConfig, setup_logging
import logging
import numpy as np

logger = logging.getLogger(__name__)


def main():
    """Enumerate and sample all Kokoro voices."""
    config = VoiceConfig()
    setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("Kokoro Voice Enumeration")
    logger.info("=" * 60)
    
    # Get all available voices
    voices = TTS.enumerate_voices()
    logger.info(f"Found {len(voices)} voices: {voices}")
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "voice_samples"
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Test phrase
    test_phrase = "Hello, I'm V. This is a voice sample for testing."
    
    # Generate sample for each voice
    tts = TTS(config)
    successful = []
    failed = []
    
    for voice in voices:
        try:
            logger.info(f"Generating sample for voice: {voice}")
            tts.set_voice(voice)
            
            audio_data, sample_rate = tts.synthesize(test_phrase)
            
            if audio_data is None or len(audio_data) == 0:
                logger.warning(f"  No audio generated for {voice}")
                failed.append(voice)
                continue
            
            # Save to WAV file
            import soundfile as sf
            output_file = output_dir / f"{voice}.wav"
            sf.write(str(output_file), audio_data, sample_rate)
            
            logger.info(f"  Saved: {output_file} ({len(audio_data)/sample_rate:.2f}s)")
            successful.append(voice)
            
        except Exception as e:
            logger.error(f"  Failed to generate sample for {voice}: {e}")
            failed.append(voice)
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Summary: {len(successful)} successful, {len(failed)} failed")
    if successful:
        logger.info(f"Successful voices: {successful}")
    if failed:
        logger.warning(f"Failed voices: {failed}")
    
    logger.info(f"Voice samples saved to: {output_dir}")
    logger.info("Open the WAV files to listen and choose your favorite voice.")
    logger.info("Then update config.yaml: tts.voice = 'your_chosen_voice'")


if __name__ == "__main__":
    main()
