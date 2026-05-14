import json
import logging
from enum import Enum
from typing import Iterator

import requests

from config_loader import VoiceConfig

logger = logging.getLogger(__name__)


class BackendType(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    LOCAL_ECHO = "local_echo"


class Backend:
    """Modular backend router. Currently supports Ollama (streaming + non-streaming)."""

    def __init__(self, config: VoiceConfig):
        self.config = config
        backend_value = config.get("backend.type", "ollama")
        try:
            self.backend_type = BackendType(backend_value)
        except ValueError:
            logger.warning(f"Unknown backend type '{backend_value}', defaulting to ollama")
            self.backend_type = BackendType.OLLAMA

        # Ollama config
        self.ollama_endpoint = config.get(
            "backend.ollama.endpoint",
            "http://localhost:11434/api/generate",
        )
        self.ollama_model = config.get("backend.ollama.model", "llama2")
        self.ollama_temperature = float(config.get("backend.ollama.temperature", 0.7))
        self.ollama_top_p = float(config.get("backend.ollama.top_p", 0.9))
        self.ollama_max_tokens = int(config.get("backend.ollama.max_tokens", 256))
        self.ollama_timeout = int(config.get("backend.ollama.timeout", 30))
        self.ollama_keep_alive = config.get("backend.ollama.keep_alive", "5m")

        # OpenAI config
        self.openai_api_key = config.get("backend.openai.api_key")
        self.openai_model = config.get("backend.openai.model", "gpt-4-turbo")

        logger.info(
            f"Backend initialized (type={self.backend_type.value}, "
            f"model={self.ollama_model if self.backend_type == BackendType.OLLAMA else self.openai_model})"
        )

    # ─── Non-streaming ─────────────────────────────────────────────────────
    def generate(self, prompt: str) -> str:
        """Generate full response (non-streaming)."""
        if self.backend_type == BackendType.LOCAL_ECHO:
            return f"Echo: {prompt}"
        if self.backend_type == BackendType.OLLAMA:
            return self._ollama_generate(prompt)
        if self.backend_type == BackendType.OPENAI:
            return self._openai_generate(prompt)
        return f"Unknown backend: {self.backend_type.value}"

    def _ollama_generate(self, prompt: str) -> str:
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.ollama_keep_alive,
            "think": False,
            "options": {
                "temperature": self.ollama_temperature,
                "top_p": self.ollama_top_p,
                "num_predict": self.ollama_max_tokens,
            },
        }
        try:
            r = requests.post(self.ollama_endpoint, json=payload, timeout=self.ollama_timeout)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Ollama connection failed (endpoint={self.ollama_endpoint}): {e}")
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")

    def _openai_generate(self, prompt: str) -> str:
        try:
            import openai
        except ImportError:
            raise RuntimeError("openai package not installed (pip install openai)")
        if not self.openai_api_key:
            raise RuntimeError("OpenAI API key not configured")
        openai.api_key = self.openai_api_key
        resp = openai.ChatCompletion.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.ollama_temperature,
            max_tokens=self.ollama_max_tokens,
        )
        return resp.choices[0].message.content.strip()

    # ─── Streaming ─────────────────────────────────────────────────────────
    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Yield response text deltas as they arrive."""
        if self.backend_type == BackendType.LOCAL_ECHO:
            for word in f"Echo: {prompt}".split():
                yield word + " "
            return
        if self.backend_type == BackendType.OLLAMA:
            yield from self._ollama_stream(prompt)
            return
        # Fallback: non-streaming as a single chunk
        yield self.generate(prompt)

    def _ollama_stream(self, prompt: str) -> Iterator[str]:
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": self.ollama_keep_alive,
            # revenant/v-9b is a thinking model; without this it spends its
            # whole token budget inside <think> and emits an empty answer.
            "think": False,
            "options": {
                "temperature": self.ollama_temperature,
                "top_p": self.ollama_top_p,
                "num_predict": self.ollama_max_tokens,
            },
        }
        # For streaming: (connect_timeout, read_timeout). Use a long read
        # timeout so cold model loads (30-90s on the first request) don't
        # kill the stream before the first token arrives.
        stream_timeout = (5, max(self.ollama_timeout, 300))
        logger.info(f"Ollama request sent (model={self.ollama_model}, prompt={prompt[:60]!r})")
        first_chunk = True
        try:
            with requests.post(
                self.ollama_endpoint,
                json=payload,
                timeout=stream_timeout,
                stream=True,
            ) as r:
                r.raise_for_status()
                logger.info("Ollama responded — waiting for tokens…")

                # Manual line-buffered NDJSON parser. requests.iter_lines can
                # buffer indefinitely with stream=True; reading raw bytes and
                # splitting on newlines guarantees per-token delivery.
                buf = b""
                # chunk_size=None tells urllib3 to yield bytes as soon as
                # they arrive, with no internal buffering.
                for raw in r.iter_content(chunk_size=None, decode_unicode=False):
                    if not raw:
                        continue
                    buf += raw
                    if b"\n" not in buf:
                        continue
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line.decode("utf-8", errors="replace"))
                        except json.JSONDecodeError:
                            logger.warning(f"Ollama returned non-JSON line: {line[:120]!r}")
                            continue
                        chunk = obj.get("response", "")
                        if chunk:
                            if first_chunk:
                                logger.info("First token received from Ollama")
                                first_chunk = False
                            yield chunk
                        if obj.get("done"):
                            return
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Ollama connection failed (endpoint={self.ollama_endpoint}): {e}")
        except Exception as e:
            raise RuntimeError(f"Ollama stream failed: {e}")

    # ─── Warmup ────────────────────────────────────────────────────────────
    def warmup(self, timeout: int = 120) -> bool:
        """
        Force the Ollama backend to load its model into VRAM now so the first
        real user turn doesn't pay the cold-load cost. No-op for echo/openai.
        Returns True on success.
        """
        if self.backend_type != BackendType.OLLAMA:
            return True
        logger.info(f"Warming up Ollama model {self.ollama_model!r} (may take up to {timeout}s)…")
        payload = {
            "model": self.ollama_model,
            "prompt": "ping",
            "stream": False,
            "keep_alive": self.ollama_keep_alive,
            "think": False,
            "options": {"num_predict": 1},
        }
        try:
            r = requests.post(self.ollama_endpoint, json=payload, timeout=(5, timeout))
            r.raise_for_status()
            logger.info(f"Ollama warmup OK (model {self.ollama_model!r} is resident)")
            return True
        except Exception as e:
            logger.warning(f"Ollama warmup failed: {e}")
            return False

    # ─── Health ────────────────────────────────────────────────────────────
    def health_check(self) -> bool:
        """Check if backend is reachable."""
        if self.backend_type == BackendType.LOCAL_ECHO:
            return True
        if self.backend_type == BackendType.OLLAMA:
            try:
                base = self.ollama_endpoint.rsplit("/api/", 1)[0]
                r = requests.get(f"{base}/api/tags", timeout=5)
                ok = r.status_code == 200
                logger.info(f"Ollama health: {'OK' if ok else 'FAILED'}")
                return ok
            except Exception as e:
                logger.warning(f"Ollama health check failed: {e}")
                return False
        if self.backend_type == BackendType.OPENAI:
            return bool(self.openai_api_key)
        return False
