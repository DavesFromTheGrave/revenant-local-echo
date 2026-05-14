import logging
import queue
import signal
import sys
import threading
import time
from typing import Optional

from config_loader import VoiceConfig, setup_logging
from audio_io import AudioIO
from wake_word import WakeWordListener
from stt import STT
from tts import TTS
from backend import Backend
from vram_manager import VRAMManager

logger = logging.getLogger(__name__)


class VoiceAssistant:
    """V — wake-word-triggered local voice assistant."""

    def __init__(self, config: VoiceConfig):
        self.config = config
        self.running = False

        self.audio = AudioIO(config)
        self.stt = STT(config)
        self.tts = TTS(config)
        self.backend = Backend(config)

        self.warm_window = float(config.get("models.warm_window", 60.0))
        self.listening_window = float(config.get("listening_window.duration", 5.0))
        self.max_speech_wait = float(config.get("listening_window.max_wait_for_speech", 5.0))
        self.max_record_duration = float(config.get("audio.max_record_duration", 30.0))

        self._warm_lock = threading.Lock()
        self._last_activity_ts: Optional[float] = None
        self._warm_thread: Optional[threading.Thread] = None
        self._warm_thread_stop = threading.Event()

        logger.info(
            f"VoiceAssistant ready (warm_window={self.warm_window}s, "
            f"listening_window={self.listening_window}s)"
        )

    # ─── Warm window management ────────────────────────────────────────────
    def _touch_activity(self):
        with self._warm_lock:
            self._last_activity_ts = time.time()
        self._ensure_warm_thread()

    def _ensure_warm_thread(self):
        if self._warm_thread and self._warm_thread.is_alive():
            return
        self._warm_thread_stop.clear()
        self._warm_thread = threading.Thread(
            target=self._warm_watchdog, daemon=True, name="warm-watchdog"
        )
        self._warm_thread.start()

    def _warm_watchdog(self):
        """Unload STT + TTS after `warm_window` seconds of no activity."""
        while not self._warm_thread_stop.is_set():
            time.sleep(1.0)
            with self._warm_lock:
                last = self._last_activity_ts
            if last is None:
                continue
            if (time.time() - last) >= self.warm_window:
                logger.info(f"Warm window expired ({self.warm_window}s) — unloading models")
                try:
                    self.stt.unload_model()
                    self.tts.unload_model()
                except Exception as e:
                    logger.error(f"Error during warm-window unload: {e}")
                with self._warm_lock:
                    self._last_activity_ts = None
                return  # watchdog exits; next activity respawns it

    # ─── Per-turn pipeline ─────────────────────────────────────────────────
    def on_wake(self):
        wake_ts = time.time()
        logger.info("Wake word fired — entering turn")
        self._touch_activity()
        try:
            self._run_turn(prewarm=True, wake_ts=wake_ts)
            self._follow_up_loop()
        except Exception as e:
            logger.error(f"Turn failed: {e}", exc_info=True)
        finally:
            self._touch_activity()

    def _run_turn(self, prewarm: bool, wake_ts: Optional[float] = None) -> bool:
        """
        Run one user→V turn. Returns True if a response was produced.
        `prewarm=True` triggers STT load on a background thread *in parallel*
        with recording, so the user can start talking immediately instead of
        waiting for Whisper to load.
        """
        preload_thread = None
        if prewarm:
            def _preload():
                try:
                    self.stt.load_model()
                except Exception as e:
                    logger.error(f"STT preload failed: {e}")
            preload_thread = threading.Thread(
                target=_preload, daemon=True, name="stt-preload"
            )
            preload_thread.start()

        if wake_ts is not None:
            logger.info(f"Opening record mic ({(time.time() - wake_ts) * 1000:.0f}ms after wake)")

        audio_data = self.audio.record_speech(
            max_wait_for_speech=self.max_speech_wait,
            max_record_duration=self.max_record_duration,
        )

        # Ensure STT preload is finished before transcribing.
        if preload_thread is not None:
            preload_thread.join(timeout=30)
        if audio_data is None:
            logger.info("No speech captured")
            return False

        try:
            transcript = self.stt.transcribe(audio_data)
        except Exception as e:
            logger.error(f"STT failed: {e}")
            return False

        if not transcript or not transcript.strip():
            logger.info("Empty transcript")
            return False

        logger.info(f"User: {transcript}")
        self._touch_activity()

        return self._stream_response(transcript)

    def _stream_response(self, user_text: str) -> bool:
        """
        Stream Ollama tokens → sentence buffer → Chatterbox per sentence → playback.

        Producer (this thread): pulls tokens from Ollama, builds sentences,
        pushes each completed sentence onto a queue.
        Consumer (speech thread): pulls sentences, synthesizes via Chatterbox,
        plays audio. Keeps Ollama draining continuously even while V is talking.
        """
        import re
        sentence_end = re.compile(r"[\.\!\?\:](?:\s|$)|\n")

        sentence_q: "queue.Queue[Optional[str]]" = queue.Queue()
        spoke_counter = {"n": 0}
        full_response_parts = []
        min_chars_to_speak = 12

        def speech_worker():
            while True:
                item = sentence_q.get()
                if item is None:
                    sentence_q.task_done()
                    return
                try:
                    self._speak_sentence(item)
                    spoke_counter["n"] += 1
                except Exception as e:
                    logger.error(f"Speech worker failed on {item!r}: {e}")
                finally:
                    sentence_q.task_done()

        worker = threading.Thread(target=speech_worker, daemon=True, name="tts-speech")
        worker.start()

        buffer = ""
        try:
            for chunk in self.backend.generate_stream(user_text):
                buffer += chunk
                full_response_parts.append(chunk)

                last_end = -1
                for m in sentence_end.finditer(buffer):
                    last_end = m.end()

                if last_end > 0 and last_end >= min_chars_to_speak:
                    to_speak = buffer[:last_end].strip()
                    buffer = buffer[last_end:]
                    if to_speak:
                        sentence_q.put(to_speak)

            tail = buffer.strip()
            if tail:
                sentence_q.put(tail)
        except Exception as e:
            logger.error(f"Backend streaming failed: {e}", exc_info=True)
            sentence_q.put(None)
            worker.join(timeout=30)
            return False

        # Signal end-of-stream and wait for playback to drain.
        sentence_q.put(None)
        worker.join()

        spoke_anything = spoke_counter["n"] > 0
        if spoke_anything:
            logger.info(f"V: {''.join(full_response_parts).strip()}")
        else:
            logger.warning("Backend produced empty response")

        return spoke_anything

    def _speak_sentence(self, sentence: str):
        try:
            audio, sr = self.tts.synthesize(sentence)
            self.audio.play_audio(audio, sample_rate=sr)
            self._touch_activity()
        except Exception as e:
            logger.error(f"TTS/playback failed for sentence {sentence!r}: {e}")

    def _follow_up_loop(self):
        """After speaking, listen for follow-ups within the listening window."""
        logger.info(f"Listening window open ({self.listening_window}s)")
        deadline = time.time() + self.listening_window

        while self.running and time.time() < deadline:
            remaining = deadline - time.time()
            wait_budget = min(remaining, self.max_speech_wait)
            if wait_budget < 0.2:
                break

            audio_data = self.audio.record_speech(
                max_wait_for_speech=wait_budget,
                max_record_duration=self.max_record_duration,
            )
            if audio_data is None:
                break

            try:
                transcript = self.stt.transcribe(audio_data)
            except Exception as e:
                logger.error(f"Follow-up STT failed: {e}")
                break

            if not transcript or not transcript.strip():
                continue

            logger.info(f"Follow-up: {transcript}")
            self._touch_activity()
            self._stream_response(transcript)
            deadline = time.time() + self.listening_window  # extend on activity

        logger.info("Listening window closed")

    # ─── Lifecycle ─────────────────────────────────────────────────────────
    def run(self):
        logger.info("Starting V…")
        self.running = True

        if VRAMManager.check_cuda_available():
            logger.info(f"CUDA OK, VRAM total: {VRAMManager.get_vram_total():.0f} MB")
        else:
            logger.warning("CUDA unavailable — STT/TTS will run on CPU (slow)")

        if not self.backend.health_check():
            logger.warning(f"Backend health check failed (type={self.backend.backend_type.value})")
        else:
            logger.info(f"Backend OK ({self.backend.backend_type.value})")
            # Synchronously warm up Ollama before arming the wake word.
            # If we did this in a background thread, a real user turn could
            # arrive while warmup is still running, and Ollama would serialize
            # the two requests — the user's request would wait behind the
            # warmup ping. Better to absorb the cold-load delay on boot.
            self.backend.warmup()

        # Instantiate wake-word listener once (model load is expensive).
        try:
            listener = WakeWordListener(self.config, on_wake=self.on_wake)
        except Exception as e:
            logger.error(f"Failed to init wake word listener: {e}", exc_info=True)
            self.shutdown()
            return

        # Main loop: arm listener → it blocks until wake fires → re-arm.
        while self.running:
            try:
                listener.listen()  # blocks until wake fires or stop()
            except KeyboardInterrupt:
                logger.info("Interrupted")
                break
            except Exception as e:
                logger.error(f"Wake word listener error: {e}", exc_info=True)
                time.sleep(1.0)

        self.shutdown()

    def shutdown(self):
        if not self.running:
            return
        logger.info("Shutting down…")
        self.running = False
        self._warm_thread_stop.set()

        try:
            self.audio.close()
        except Exception as e:
            logger.error(f"Error closing audio: {e}")
        try:
            self.stt.unload_model()
        except Exception as e:
            logger.error(f"Error unloading STT: {e}")
        try:
            self.tts.unload_model()
        except Exception as e:
            logger.error(f"Error unloading TTS: {e}")
        VRAMManager.clear_cache()
        logger.info("Shutdown complete")


def main():
    config = VoiceConfig()
    setup_logging(config)

    logger.info("=" * 60)
    logger.info("Revenant Echo — V (wake-word-only)")
    logger.info("=" * 60)

    assistant = VoiceAssistant(config)

    def _signal_handler(_sig, _frame):
        logger.info("Signal received, shutting down…")
        assistant.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
    except (AttributeError, ValueError):
        # SIGTERM not always available on Windows main thread
        pass

    assistant.run()


if __name__ == "__main__":
    main()
