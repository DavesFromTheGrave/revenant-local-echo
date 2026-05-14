import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pyaudio

from config_loader import VoiceConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)


class WakeWordListener:
    """
    OpenWakeWord detector running on CPU, always-listening.
    Opens its own PyAudio input stream (does not share AudioIO's, to avoid
    contention with the recording stream used after the wake word fires).
    Triggers callback on wake word detection.

    The `wake_word.model` config value can be either:
      * A built-in OpenWakeWord model name ("alexa", "hey_jarvis", etc.)
      * A path to a custom ONNX model — absolute, or relative to project root
        (e.g. "models/hey_friday.onnx")
    """

    def __init__(self, config: VoiceConfig, on_wake: Optional[Callable] = None):
        self.config = config
        self.on_wake = on_wake
        self.is_listening = False
        self._stream = None
        self._pa = None

        model_spec = config.get("wake_word.model", "alexa")
        self.threshold = float(config.get("wake_word.threshold", 0.5))

        # If the spec looks like a path (contains / \ or ends in .onnx/.tflite),
        # resolve it. Otherwise pass through as a built-in name.
        resolved = model_spec
        if any(s in model_spec for s in ("/", "\\")) or model_spec.endswith((".onnx", ".tflite")):
            p = Path(model_spec)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if not p.exists():
                raise FileNotFoundError(f"Wake word model not found: {p}")
            resolved = str(p)

        logger.info(f"Loading OpenWakeWord model: {resolved}")
        import openwakeword
        self.model = openwakeword.Model(
            wakeword_models=[resolved],
            inference_framework="onnx",
        )

    def _open_stream(self):
        sample_rate = int(self.config.get("audio.sample_rate", 16000))
        chunk_size = int(self.config.get("audio.chunk_size", 1024))
        device_id = self.config.get("audio.input_device")

        self._pa = pyaudio.PyAudio()
        if device_id is None:
            device_id = self._pa.get_default_input_device_info()["index"]

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=int(device_id),
            frames_per_buffer=chunk_size,
        )
        return sample_rate, chunk_size

    def _close_stream(self):
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None
        finally:
            if self._pa is not None:
                self._pa.terminate()
                self._pa = None

    def listen(self):
        """Blocking listen loop. Returns when wake word fires or stop() is called."""
        logger.info("Wake word listener started")
        self.is_listening = True

        # Reset the model's internal prediction buffer. Without this, residual
        # state from a prior detection causes instant false triggers on re-arm.
        try:
            if hasattr(self.model, "reset"):
                self.model.reset()
        except Exception as e:
            logger.warning(f"Wake model reset failed: {e}")

        try:
            sample_rate, chunk_size = self._open_stream()

            # Tiny warmup discard (~0.1s) — just enough to skip driver
            # cold-start spikes without eating the front of an "Alexa" that
            # comes in fast. The model.reset() above is what actually prevents
            # the residual-state false trigger.
            warmup_chunks = max(1, int(0.1 * sample_rate / chunk_size))
            for _ in range(warmup_chunks):
                self._stream.read(chunk_size, exception_on_overflow=False)

            while self.is_listening:
                data = self._stream.read(chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)

                predictions = self.model.predict(audio_chunk)

                for name, score in predictions.items():
                    if score > self.threshold:
                        logger.info(f"Wake word detected! ({name}: {score:.3f})")
                        self._close_stream()
                        if self.on_wake is not None:
                            self.on_wake()
                        return
        except KeyboardInterrupt:
            logger.info("Wake word listener interrupted")
        except Exception as e:
            logger.error(f"Wake word listener error: {e}", exc_info=True)
        finally:
            self.is_listening = False
            self._close_stream()

    def stop(self):
        """Stop listening."""
        self.is_listening = False
        logger.info("Wake word listener stopped")
