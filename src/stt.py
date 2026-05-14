import logging

import numpy as np

from config_loader import VoiceConfig
from vram_manager import VRAMManager

logger = logging.getLogger(__name__)


class STT:
    """Faster-Whisper STT with CUDA support, VAD tuning, and lazy load/unload."""

    def __init__(self, config: VoiceConfig):
        self.config = config
        self.model = None
        # Do not load at construction — caller controls warm-window lifecycle.

    def load_model(self):
        """Load Whisper model if not already loaded."""
        if self.model is not None:
            return

        from faster_whisper import WhisperModel

        model_size = self.config.get("stt.model_size", "medium")
        device = self.config.get("stt.device", "cuda")
        compute_type = self.config.get("stt.compute_type", "float16")

        logger.info(f"Loading Whisper {model_size} (device={device}, compute={compute_type})")
        VRAMManager.log_vram_snapshot("before STT load")

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            local_files_only=False,
        )

        VRAMManager.log_vram_snapshot("after STT load")

    def unload_model(self):
        """Unload model and clear VRAM."""
        if self.model is None:
            return
        del self.model
        self.model = None
        VRAMManager.clear_cache("STT")
        logger.debug("STT model unloaded")

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio with prompt bias and VAD tuning.
        Auto-loads the model if not already loaded.
        """
        self.load_model()

        language = self.config.get("stt.language", "en")
        initial_prompt = self.config.get("stt.initial_prompt", "")
        no_speech_threshold = float(self.config.get("stt.no_speech_threshold", 0.3))
        # Whisper VAD silence-to-segment-boundary value (ms).
        # 500ms is a sensible default; we end the *recording* with our own VAD
        # in audio_io.record_speech, so this only affects internal segmentation.
        vad_min_silence_ms = int(self.config.get("stt.vad_min_silence_ms", 500))

        if audio_data is None or len(audio_data) == 0:
            return ""

        # Resample-or-error: faster-whisper expects 16kHz mono float32.
        if sample_rate != 16000:
            logger.warning(f"Audio sample_rate={sample_rate}, expected 16000 — results may degrade")

        logger.debug(
            f"Transcribing {len(audio_data) / sample_rate:.1f}s audio "
            f"(no_speech_threshold={no_speech_threshold})"
        )

        # Our own VAD in audio_io.record_speech already decided where speech
        # starts/stops. Whisper's internal vad_filter is a *second* VAD that
        # re-judges the clipped audio and was discarding quiet speech entirely.
        # Disable it — transcribe whatever record_speech captured.
        segments, _info = self.model.transcribe(
            audio_data,
            language=language,
            initial_prompt=initial_prompt or None,
            no_speech_threshold=no_speech_threshold,
            vad_filter=False,
        )

        transcript = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info(f"Transcribed: {transcript!r}")
        return transcript
