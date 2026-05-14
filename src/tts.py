import logging
from typing import Iterator, Iterable, Tuple

import numpy as np

from config_loader import VoiceConfig
from vram_manager import VRAMManager

logger = logging.getLogger(__name__)


class TTS:
    """
    Kokoro TTS wrapper (Kokoro 0.9.x API: KPipeline + KModel).
    Externally-controlled load/unload lifecycle so callers manage warm window.
    """

    def __init__(self, config: VoiceConfig):
        self.config = config
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.device = "cpu"

        self.pipeline = None
        self.voice = config.get("tts.voice", "af_nicole")
        self.sample_rate = int(config.get("tts.sample_rate", 24000))
        self.speed = float(config.get("tts.speed", 1.25))
        self.lang_code = config.get("tts.lang_code", "a")  # 'a' = American English

        logger.info(
            f"TTS initialized (device={self.device}, voice={self.voice}, "
            f"speed={self.speed}, lang={self.lang_code})"
        )

    def load_model(self):
        """Load Kokoro pipeline into VRAM if not already loaded."""
        if self.pipeline is not None:
            return

        VRAMManager.clear_cache()
        before = VRAMManager.get_vram_available()

        from kokoro import KPipeline
        self.pipeline = KPipeline(lang_code=self.lang_code, device=self.device)

        after = VRAMManager.get_vram_available()
        used = before - after
        logger.info(f"Kokoro loaded (voice={self.voice}, VRAM used={used:.2f}GB)")

    def unload_model(self):
        """Unload Kokoro from VRAM."""
        if self.pipeline is None:
            return
        try:
            del self.pipeline
            self.pipeline = None
            VRAMManager.clear_cache("TTS")
            logger.info(f"Kokoro unloaded (VRAM available={VRAMManager.get_vram_available():.2f}GB)")
        except Exception as e:
            logger.error(f"Failed to unload Kokoro: {e}")

    def _synthesize_text(self, text: str) -> np.ndarray:
        """Synthesize one chunk of text → float32 numpy audio in [-1, 1]."""
        import torch

        self.load_model()

        audio_chunks = []
        with torch.no_grad():
            for result in self.pipeline(text, voice=self.voice, speed=self.speed):
                if result.audio is None:
                    continue
                chunk = result.audio
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.detach().cpu().numpy()
                else:
                    chunk = np.asarray(chunk)
                audio_chunks.append(chunk)

        if not audio_chunks:
            logger.warning(f"Kokoro produced no audio for text {text!r}")
            return np.zeros((self.sample_rate,), dtype=np.float32)

        audio = np.concatenate(audio_chunks).astype(np.float32, copy=False)

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / (peak + 1e-8)

        return audio

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize whole text in one shot. Returns (float32 audio, sample_rate)."""
        if not text or not text.strip():
            return np.zeros((self.sample_rate,), dtype=np.float32), self.sample_rate

        audio = self._synthesize_text(text.strip())
        logger.debug(f"Synthesized {len(audio)} samples ({len(audio)/self.sample_rate:.2f}s)")
        return audio, self.sample_rate

    def synthesize_sentences(self, sentences: Iterable[str]) -> Iterator[Tuple[np.ndarray, int]]:
        """Yield (audio, sample_rate) per sentence — for streaming callers."""
        for sentence in sentences:
            text = sentence.strip()
            if not text:
                continue
            audio = self._synthesize_text(text)
            yield audio, self.sample_rate
