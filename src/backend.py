import json
import logging
import os
from enum import Enum
from typing import Iterator

import requests
from dotenv import load_dotenv

from config_loader import VoiceConfig

logger = logging.getLogger(__name__)


class BackendType(Enum):
    OLLAMA = "ollama"
    MEGALLM = "megallm"
    OPENAI = "openai"
    LOCAL_ECHO = "local_echo"


class Backend:
    """Modular backend router.

    Supports:
    - Ollama local generation
    - MegaLLM OpenAI-compatible cloud generation
    - OpenAI legacy backend
    - Local echo test backend
    """

    def __init__(self, config: VoiceConfig):
        load_dotenv()

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

        # Shared voice brevity instruction.
        self.brevity_prefix = config.get(
            "backend.brevity_prefix",
            "[Voice mode: respond in 1-2 short sentences. No preamble, no follow-up offers, "
            "no explanation unless asked. Be conversational and direct.]\n\nUser: ",
        )

        # MegaLLM config. Uses OpenAI-compatible /chat/completions.
        self.megallm_base_url = str(
            config.get("backend.megallm.base_url", "https://ai.megallm.io/v1")
        ).rstrip("/")
        self.megallm_model = config.get("backend.megallm.model", "grok-4.1-fast-reasoning")
        self.megallm_api_key_env = config.get("backend.megallm.api_key_env", "MEGALLM_API_KEY")
        self.megallm_api_key = os.getenv(self.megallm_api_key_env, "")
        self.megallm_temperature = float(config.get("backend.megallm.temperature", 0.7))
        self.megallm_top_p = float(config.get("backend.megallm.top_p", 0.9))
        self.megallm_max_tokens = int(config.get("backend.megallm.max_tokens", 120))
        self.megallm_timeout = int(config.get("backend.megallm.timeout", 60))

        # Legacy OpenAI config
        self.openai_api_key = config.get("backend.openai.api_key")
        self.openai_model = config.get("backend.openai.model", "gpt-4-turbo")

        active_model = {
            BackendType.OLLAMA: self.ollama_model,
            BackendType.MEGALLM: self.megallm_model,
            BackendType.OPENAI: self.openai_model,
            BackendType.LOCAL_ECHO: "local_echo",
        }.get(self.backend_type, "unknown")

        logger.info(f"Backend initialized (type={self.backend_type.value}, model={active_model})")

    # ─── Non-streaming ─────────────────────────────────────────────────────
    def generate(self, prompt: str) -> str:
        """Generate full response (non-streaming)."""
        if self.backend_type == BackendType.LOCAL_ECHO:
            return f"Echo: {prompt}"
        if self.backend_type == BackendType.OLLAMA:
            return self._ollama_generate(prompt)
        if self.backend_type == BackendType.MEGALLM:
            return self._megallm_generate(prompt)
        if self.backend_type == BackendType.OPENAI:
            return self._openai_generate(prompt)

        return f"Unknown backend: {self.backend_type.value}"

    def _ollama_generate(self, prompt: str) -> str:
        wrapped = f"{self.brevity_prefix}{prompt}"
        payload = {
            "model": self.ollama_model,
            "prompt": wrapped,
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

    def _megallm_headers(self) -> dict[str, str]:
        if not self.megallm_api_key:
            raise RuntimeError(
                f"MegaLLM API key not configured. Set {self.megallm_api_key_env} in .env"
            )

        return {
            "Authorization": f"Bearer {self.megallm_api_key}",
            "Content-Type": "application/json",
        }

    def _megallm_payload(self, prompt: str, stream: bool) -> dict:
        return {
            "model": self.megallm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are V speaking in voice mode. Respond in 1-2 short sentences. "
                        "No preamble. No follow-up offers. Be conversational and direct."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.megallm_temperature,
            "top_p": self.megallm_top_p,
            "max_tokens": self.megallm_max_tokens,
            "stream": stream,
        }

    def _megallm_generate(self, prompt: str) -> str:
        url = f"{self.megallm_base_url}/chat/completions"
        payload = self._megallm_payload(prompt, stream=False)

        try:
            r = requests.post(
                url,
                headers=self._megallm_headers(),
                json=payload,
                timeout=self.megallm_timeout,
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"MegaLLM connection failed (endpoint={url}): {e}")
        except requests.exceptions.HTTPError as e:
            body = e.response.text[:500] if e.response is not None else ""
            raise RuntimeError(f"MegaLLM HTTP error: {e}. Body: {body}")
        except Exception as e:
            raise RuntimeError(f"MegaLLM request failed: {e}")

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

        if self.backend_type == BackendType.MEGALLM:
            yield from self._megallm_stream(prompt)
            return

        # Fallback: non-streaming as a single chunk
        yield self.generate(prompt)

    def _ollama_stream(self, prompt: str) -> Iterator[str]:
        wrapped = f"{self.brevity_prefix}{prompt}"
        payload = {
            "model": self.ollama_model,
            "prompt": wrapped,
            "stream": True,
            "keep_alive": self.ollama_keep_alive,
            "think": False,
            "options": {
                "temperature": self.ollama_temperature,
                "top_p": self.ollama_top_p,
                "num_predict": self.ollama_max_tokens,
            },
        }

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

                buf = b""

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

    def _megallm_stream(self, prompt: str) -> Iterator[str]:
        url = f"{self.megallm_base_url}/chat/completions"
        payload = self._megallm_payload(prompt, stream=True)
        stream_timeout = (5, max(self.megallm_timeout, 300))

        logger.info(f"MegaLLM request sent (model={self.megallm_model}, prompt={prompt[:60]!r})")

        first_chunk = True

        try:
            with requests.post(
                url,
                headers=self._megallm_headers(),
                json=payload,
                timeout=stream_timeout,
                stream=True,
            ) as r:
                r.raise_for_status()
                logger.info("MegaLLM responded — waiting for tokens…")

                for raw_line in r.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue

                    line = raw_line.strip()

                    if not line.startswith("data:"):
                        continue

                    data = line[len("data:"):].strip()

                    if data == "[DONE]":
                        return

                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning(f"MegaLLM returned non-JSON stream line: {line[:120]!r}")
                        continue

                    choices = obj.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    chunk = delta.get("content", "")

                    if chunk:
                        if first_chunk:
                            logger.info("First token received from MegaLLM")
                            first_chunk = False
                        yield chunk

        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"MegaLLM connection failed (endpoint={url}): {e}")
        except requests.exceptions.HTTPError as e:
            body = e.response.text[:500] if e.response is not None else ""
            raise RuntimeError(f"MegaLLM HTTP error: {e}. Body: {body}")
        except Exception as e:
            raise RuntimeError(f"MegaLLM stream failed: {e}")

    # ─── Warmup ────────────────────────────────────────────────────────────
    def warmup(self, timeout: int = 120) -> bool:
        """
        Force the Ollama backend to load its model into VRAM now so the first
        real user turn doesn't pay the cold-load cost. No-op for cloud/echo/openai.
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

        if self.backend_type == BackendType.MEGALLM:
            try:
                url = f"{self.megallm_base_url}/models"
                r = requests.get(url, headers=self._megallm_headers(), timeout=10)
                ok = r.status_code == 200
                logger.info(f"MegaLLM health: {'OK' if ok else 'FAILED'}")
                return ok
            except Exception as e:
                logger.warning(f"MegaLLM health check failed: {e}")
                return False

        if self.backend_type == BackendType.OPENAI:
            return bool(self.openai_api_key)

        return False