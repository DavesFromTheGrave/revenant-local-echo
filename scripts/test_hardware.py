"""
Test hardware configuration (CUDA, audio devices, VRAM).
Run this before first use to verify your system is ready.

Usage:
    python test_hardware.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config_loader import VoiceConfig, setup_logging
from vram_manager import VRAMManager
from audio_io import AudioIO
import logging

logger = logging.getLogger(__name__)


def test_cuda():
    """Test CUDA availability and VRAM."""
    logger.info("-" * 60)
    logger.info("CUDA Test")
    logger.info("-" * 60)
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        
        is_available = VRAMManager.is_cuda_available()
        logger.info(f"CUDA available: {is_available}")
        
        if is_available:
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            total = VRAMManager.get_vram_total()
            available = VRAMManager.get_vram_available()
            logger.info(f"VRAM total: {total:.1f}MB")
            logger.info(f"VRAM available: {available:.1f}MB")
            
            if available < 2048:
                logger.warning("WARNING: Less than 2GB VRAM available (may be tight)")
            else:
                logger.info("✓ VRAM sufficient for pipeline")
        else:
            logger.warning("CUDA not available - will use CPU (slower)")
    
    except Exception as e:
        logger.error(f"CUDA test failed: {e}")


def test_audio():
    """Test audio device enumeration."""
    logger.info("-" * 60)
    logger.info("Audio Devices Test")
    logger.info("-" * 60)
    
    try:
        config = VoiceConfig()
        audio = AudioIO(config)
        
        devices = audio.enumerate_devices()
        logger.info(f"Total audio devices: {len(devices)}")
        
        for idx, device_info in devices.items():
            device_type = "[INPUT]" if device_info.get("max_input_channels", 0) > 0 else "[OUTPUT]"
            logger.info(
                f"  {idx}: {device_info.get('name')} {device_type} "
                f"(channels={device_info.get('max_input_channels', device_info.get('max_output_channels', 0))})"
            )
        
        input_device = audio.get_input_device()
        output_device = audio.get_output_device()
        logger.info(f"Selected input: {input_device}")
        logger.info(f"Selected output: {output_device}")
        
        logger.info("✓ Audio devices detected")
        audio.close()
        
    except Exception as e:
        logger.error(f"Audio test failed: {e}")


def test_models():
    """Test model availability (faster-whisper, OpenWakeWord, Chatterbox)."""
    logger.info("-" * 60)
    logger.info("Model Availability Test")
    logger.info("-" * 60)

    # faster-whisper
    try:
        from faster_whisper import WhisperModel  # noqa: F401
        logger.info("✓ faster-whisper available")
    except ImportError:
        logger.warning("✗ faster-whisper not available (pip install faster-whisper)")

    # OpenWakeWord
    try:
        from openwakeword.model import Model  # noqa: F401
        logger.info("✓ OpenWakeWord available")
    except ImportError:
        logger.warning("✗ OpenWakeWord not available (pip install openwakeword)")

    # Chatterbox
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: F401
        logger.info("✓ Chatterbox Turbo available")
    except ImportError:
        logger.warning("✗ Chatterbox not available (pip install chatterbox-tts)")
    
    # PyAudio
    try:
        import pyaudio
        logger.info("✓ PyAudio available")
    except ImportError:
        logger.warning("✗ PyAudio not available (pip install pyaudio)")
    
    # keyboard
    try:
        import keyboard
        logger.info("✓ keyboard available (for PTT)")
    except ImportError:
        logger.warning("✗ keyboard not available (pip install keyboard)")


def test_backend():
    """Test backend connectivity."""
    logger.info("-" * 60)
    logger.info("Backend Test")
    logger.info("-" * 60)
    
    try:
        from backend import Backend
        config = VoiceConfig()
        backend = Backend(config)
        
        logger.info(f"Backend type: {backend.backend_type.value}")
        
        is_healthy = backend.health_check()
        if is_healthy:
            logger.info("✓ Backend health check passed")
        else:
            logger.warning("✗ Backend health check failed (check endpoint)")
            if backend.backend_type.value == "ollama":
                logger.info("  Hint: Start Ollama with 'ollama serve'")
        
    except Exception as e:
        logger.error(f"Backend test failed: {e}")


def main():
    """Run all hardware tests."""
    config = VoiceConfig()
    setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("Revenant Echo - Hardware Test")
    logger.info("=" * 60)
    logger.info("")
    
    test_cuda()
    logger.info("")
    
    test_audio()
    logger.info("")
    
    test_models()
    logger.info("")
    
    test_backend()
    logger.info("")
    
    logger.info("=" * 60)
    logger.info("Hardware test complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
