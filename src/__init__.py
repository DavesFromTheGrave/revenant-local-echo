"""Revenant Echo - Local Voice Assistant Pipeline"""

__version__ = "0.1.0"

from .config_loader import VoiceConfig, setup_logging
from .vram_manager import VRAMManager
from .audio_io import AudioIO
from .wake_word import WakeWordListener
from .stt import STT
from .tts import TTS
from .backend import Backend

__all__ = [
    "VoiceConfig",
    "setup_logging",
    "VRAMManager",
    "AudioIO",
    "WakeWordListener",
    "STT",
    "TTS",
    "Backend",
]
