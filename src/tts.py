import logging
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import numpy as np

from config_loader import VoiceConfig, PROJECT_ROOT
from vram_manager import VRAMManager

logger = logging.getLogger(__name__)


class TTS:
    """
    Chatterbox Turbo TTS wrapper with zero-shot voice cloning.

    On load:
      * Pulls the Chatterbox Turbo model into VRAM.
      * Calls `prepare_conditionals()` ONCE against the configured voice
        reference WAV, so every subsequent `generate()` skips re-conditioning
        and runs ~2x faster.

    Externally-controlled load/unload lifecycle — main.py's warm-window
    watchdog calls unload_model() after idle.
    """

    def __init__(self, config: VoiceConfig):
        self.config = config

        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.device = "cpu"

        self.model = None

        # ─── Config knobs ─────────────────────────────────────────────────
        voice_ref = config.get("tts.voice_ref", "models/voice_refs/v_combined.wav")
        ref_path = Path(voice_ref)
        if not ref_path.is_absolute():
            ref_path = PROJECT_ROOT / ref_path
        self.voice_ref_path = ref_path

        # Turbo only respects: text, temperature, top_k, top_p,
        # repetition_penalty. cfg_weight / min_p / exaggeration are warned
        # about and ignored, so don't expose them.
        self.temperature = float(config.get("tts.temperature", 0.8))
        self.top_p = float(config.get("tts.top_p", 0.95))
        self.top_k = int(config.get("tts.top_k", 1000))
        self.repetition_penalty = float(config.get("tts.repetition_penalty", 1.2))
        self.norm_loudness = bool(config.get("tts.norm_loudness", True))

        # Chatterbox Turbo's internal sample rate (24kHz). Overwritten when
        # the model loads — we read it from the loaded instance.
        self.sample_rate = 24000

        logger.info(
            f"TTS initialized (Chatterbox Turbo, device={self.device}, "
            f"voice_ref={self.voice_ref_path.name})"
        )

    # ─── Lifecycle ────────────────────────────────────────────────────────
    def load_model(self):
        """Load Chatterbox into VRAM and pre-compute voice conditionals."""
        if self.model is not None:
            return

        if not self.voice_ref_path.exists():
            raise FileNotFoundError(
                f"Voice reference WAV not found: {self.voice_ref_path}. "
                f"Drop a 10-30s clip of the target voice there, or update "
                f"tts.voice_ref in config.yaml."
            )

        VRAMManager.clear_cache()
        before = VRAMManager.get_vram_available()

        from chatterbox.tts_turbo import ChatterboxTurboTTS
        self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)
        # Cache the conditionals from the reference voice ONCE so every
        # subsequent generate() can skip the librosa load + ref embedding.
        self.model.prepare_conditionals(
            str(self.voice_ref_path),
            norm_loudness=self.norm_loudness,
        )
        # Trust the model's own sample rate.
        self.sample_rate = int(self.model.sr)

        after = VRAMManager.get_vram_available()
        used = before - after
        logger.info(
            f"Chatterbox loaded (sr={self.sample_rate}, "
            f"VRAM used={used:.2f}GB, ref={self.voice_ref_path.name})"
        )

    def unload_model(self):
        """Unload Chatterbox from VRAM."""
        if self.model is None:
            return
        try:
            del self.model
            self.model = None
            VRAMManager.clear_cache("TTS")
            logger.info(
                f"Chatterbox unloaded (VRAM available="
                f"{VRAMManager.get_vram_available():.2f}GB)"
            )
        except Exception as e:
            logger.error(f"Failed to unload Chatterbox: {e}")

    # ─── Synthesis ────────────────────────────────────────────────────────
    def _synthesize_text(self, text: str) -> np.ndarray:
        """Synthesize one chunk → float32 numpy audio in [-1, 1]."""
        import torch

        self.load_model()

        with torch.no_grad():
            # audio_prompt_path=None → reuse cached conditionals from load_model().
            wav = self.model.generate(
                text,
                audio_prompt_path=None,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                norm_loudness=self.norm_loudness,
            )

        # Chatterbox returns torch.FloatTensor [1, samples]. Squeeze to 1D.
        if isinstance(wav, torch.Tensor):
            audio = wav.detach().cpu().numpy()
        else:
            audio = np.asarray(wav)

        if audio.ndim > 1:
            audio = np.squeeze(audio)

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / (peak + 1e-8)

        return audio

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize whole text in one shot. Returns (float32 audio, sample_rate)."""
        if not text or not text.strip():
            return np.zeros((self.sample_rate,), dtype=np.float32), self.sample_rate

        audio = self._synthesize_text(text.strip())
        logger.debug(
            f"Synthesized {len(audio)} samples "
            f"({len(audio) / self.sample_rate:.2f}s)"
        )
        return audio, self.sample_rate

    def synthesize_sentences(
        self, sentences: Iterable[str]
    ) -> Iterator[Tuple[np.ndarray, int]]:
        """Yield (audio, sample_rate) per sentence — for streaming callers."""
        for sentence in sentences:
            text = sentence.strip()
            if not text:
                continue
            audio = self._synthesize_text(text)
            yield audio, self.sample_rate
