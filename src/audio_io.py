import logging
import time
from typing import Optional, Union

import numpy as np
import pyaudio

from config_loader import VoiceConfig

logger = logging.getLogger(__name__)


def _resolve_device(p: pyaudio.PyAudio, spec: Union[int, str, None], want_input: bool) -> Optional[int]:
    """
    Resolve an audio device from config.

    - None  → PyAudio's default device of the requested direction
    - int   → used directly as the device index
    - str   → case-insensitive substring match against device names. First
              device whose name contains the substring AND has the right
              direction (input/output) wins. Falls back to default on miss.
    """
    direction = "input" if want_input else "output"
    default = p.get_default_input_device_info() if want_input else p.get_default_output_device_info()

    if spec is None:
        return int(default["index"])

    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        return int(spec)

    needle = str(spec).strip().lower()
    matches = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        channels_key = "maxInputChannels" if want_input else "maxOutputChannels"
        if info.get(channels_key, 0) <= 0:
            continue
        if needle in info["name"].lower():
            matches.append((i, info["name"]))

    if not matches:
        logger.warning(f"No {direction} device matched name={spec!r}; falling back to default {default['index']}")
        return int(default["index"])

    chosen_idx, chosen_name = matches[0]
    if len(matches) > 1:
        logger.info(
            f"Multiple {direction} devices matched name={spec!r}; picking first: "
            f"[{chosen_idx}] {chosen_name!r} (others: {[m[1] for m in matches[1:]]})"
        )
    else:
        logger.info(f"Resolved {direction} device name={spec!r} → [{chosen_idx}] {chosen_name!r}")
    return chosen_idx


def resolve_input_device(p: pyaudio.PyAudio, spec: Union[int, str, None]) -> Optional[int]:
    return _resolve_device(p, spec, want_input=True)


def resolve_output_device(p: pyaudio.PyAudio, spec: Union[int, str, None]) -> Optional[int]:
    return _resolve_device(p, spec, want_input=False)


class AudioIO:
    """Audio input/output management with VAD-based speech recording."""

    def __init__(self, config: VoiceConfig):
        self.config = config
        self.p = pyaudio.PyAudio()
        self._enumerate_devices()

    def _enumerate_devices(self):
        """List all audio devices."""
        logger.info("Available audio devices:")
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            logger.info(
                f"  [{i}] {info['name']} | "
                f"In: {info['maxInputChannels']} | "
                f"Out: {info['maxOutputChannels']}"
            )

    def enumerate_devices(self) -> dict:
        """Return a dictionary of all audio devices."""
        devices = {}
        for i in range(self.p.get_device_count()):
            devices[i] = self.p.get_device_info_by_index(i)
        return devices

    def get_input_device(self) -> Optional[int]:
        """Resolve the input device from config (int index or name substring)."""
        return resolve_input_device(self.p, self.config.get("audio.input_device"))

    def get_output_device(self) -> Optional[int]:
        """Resolve the output device from config (int index or name substring)."""
        return resolve_output_device(self.p, self.config.get("audio.output_device"))

    def play_audio(self, audio_data: np.ndarray, sample_rate: int = 24000):
        """Play float32 audio to output device."""
        device_id = self.get_output_device()
        chunk_size = int(self.config.get("audio.chunk_size", 1024))

        logger.debug(f"Playing audio: {len(audio_data)} samples @ {sample_rate}Hz")

        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            output=True,
            output_device_index=device_id,
            frames_per_buffer=chunk_size,
        )

        try:
            audio_bytes = audio_data.astype(np.float32).tobytes()
            stream.write(audio_bytes)
        finally:
            stream.stop_stream()
            stream.close()

        logger.debug("Audio playback finished")

    def record_speech(
        self,
        max_wait_for_speech: float = 5.0,
        max_record_duration: float = 30.0,
    ) -> Optional[np.ndarray]:
        """
        Record speech with proper VAD lifecycle:
          1. Wait up to `max_wait_for_speech` seconds for first speech frame.
          2. Once speech starts, record until `silence_duration` of trailing
             silence is detected, or until `max_record_duration` is reached.

        Returns float32 numpy array of recorded audio, or None if no speech
        ever started within the wait window.
        """
        sample_rate = int(self.config.get("audio.sample_rate", 16000))
        chunk_size = int(self.config.get("audio.chunk_size", 1024))
        silence_duration = float(self.config.get("stt.silence_duration", 2.0))
        rms_threshold = float(self.config.get("audio.vad_rms_threshold", 0.02))
        device_id = self.get_input_device()

        chunks_per_sec = sample_rate / chunk_size
        silence_chunks_needed = max(1, int(silence_duration * chunks_per_sec))

        logger.debug(
            f"record_speech: wait≤{max_wait_for_speech}s, max≤{max_record_duration}s, "
            f"silence_to_stop={silence_duration}s, rms_threshold={rms_threshold}"
        )

        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=device_id,
            frames_per_buffer=chunk_size,
        )

        frames = []
        speech_started = False
        silence_chunks = 0
        wait_start = time.time()
        speech_start_time = None
        rms_samples = []  # collected while waiting for speech, for diagnostics

        try:
            while True:
                data = stream.read(chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                rms = float(np.sqrt(np.mean(audio_chunk ** 2))) if audio_chunk.size else 0.0

                if not speech_started:
                    rms_samples.append(rms)
                    if rms >= rms_threshold:
                        speech_started = True
                        speech_start_time = time.time()
                        frames.append(audio_chunk)
                        logger.info(f"Speech started (rms={rms:.4f})")
                    else:
                        if (time.time() - wait_start) > max_wait_for_speech:
                            if rms_samples:
                                peak = max(rms_samples)
                                avg = sum(rms_samples) / len(rms_samples)
                                logger.info(
                                    f"No speech within wait window "
                                    f"(threshold={rms_threshold}, peak_rms={peak:.4f}, "
                                    f"avg_rms={avg:.4f}, samples={len(rms_samples)})"
                                )
                            else:
                                logger.info("No speech within wait window (no audio chunks read)")
                            return None
                        continue
                else:
                    frames.append(audio_chunk)

                    if rms < rms_threshold:
                        silence_chunks += 1
                    else:
                        silence_chunks = 0

                    if silence_chunks >= silence_chunks_needed:
                        logger.debug(f"Trailing silence reached ({silence_duration}s)")
                        break

                    if speech_start_time is not None and \
                            (time.time() - speech_start_time) > max_record_duration:
                        logger.debug(f"Max record duration reached ({max_record_duration}s)")
                        break
        except Exception as e:
            logger.error(f"record_speech error: {e}", exc_info=True)
        finally:
            stream.stop_stream()
            stream.close()

        if not frames:
            return None

        audio = np.concatenate(frames)
        logger.info(f"Recorded {len(audio)} samples ({len(audio)/sample_rate:.2f}s)")
        return audio

    def close(self):
        """Clean up PyAudio."""
        try:
            self.p.terminate()
        except Exception as e:
            logger.error(f"AudioIO close error: {e}")
        logger.info("AudioIO closed")
