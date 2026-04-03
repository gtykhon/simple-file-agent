"""Multi-model LLM client.

Strategy: use Ollama (local, free) by default.
Automatically falls back to Anthropic Claude on connection failure or rate limit (HTTP 429).
Rate-limit cooldown: 1 hour before retrying Ollama.
"""

import logging
import os
import time
from typing import Tuple

import httpx

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Ollama                                                                        #
# --------------------------------------------------------------------------- #

class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system: str = "") -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
        }
        r = httpx.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        if r.status_code == 429:
            raise RuntimeError("RATE_LIMITED")
        r.raise_for_status()
        return r.json()["response"].strip()


# --------------------------------------------------------------------------- #
# Anthropic Claude                                                               #
# --------------------------------------------------------------------------- #

class ClaudeClient:
    DEFAULT_SYSTEM = "You are a precise coding assistant. Follow instructions exactly."

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        try:
            import anthropic
            self._anthropic = anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")

        self.client = self._anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system: str = "") -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system or self.DEFAULT_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()


# --------------------------------------------------------------------------- #
# Unified multi-model client                                                    #
# --------------------------------------------------------------------------- #

class MultiModelClient:
    """
    Tries Ollama first (local). Falls back to Claude on:
      - Ollama not running
      - Connection error
      - Rate limit (HTTP 429) — suppresses Ollama for 1 hour

    Usage:
        client = MultiModelClient()
        response, model_used = client.generate("Write a hello world function")
    """

    RATE_LIMIT_COOLDOWN = 3600  # seconds

    def __init__(
        self,
        ollama_model: str = "llama3.1",
        claude_model: str = "claude-haiku-4-5-20251001",
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.ollama = OllamaClient(base_url=ollama_base_url, model=ollama_model)
        self.claude = ClaudeClient(model=claude_model)
        self._ollama_blocked_until: float = 0.0

    def _ollama_available(self) -> bool:
        if time.time() < self._ollama_blocked_until:
            logger.debug("Ollama in cooldown, skipping")
            return False
        return self.ollama.is_available()

    def generate(
        self,
        prompt: str,
        system: str = "",
        force_cloud: bool = False,
    ) -> Tuple[str, str]:
        """
        Returns (response_text, model_name).

        Args:
            prompt: User prompt.
            system: Optional system prompt override.
            force_cloud: Skip Ollama and go straight to Claude.
        """
        if not force_cloud and self._ollama_available():
            try:
                response = self.ollama.generate(prompt, system)
                return response, f"ollama/{self.ollama.model}"
            except RuntimeError as e:
                if "RATE_LIMITED" in str(e):
                    self._ollama_blocked_until = time.time() + self.RATE_LIMIT_COOLDOWN
                    logger.warning("Ollama rate limited — cooling down for 1h, falling back to Claude")
                else:
                    logger.warning("Ollama error: %s — falling back to Claude", e)
            except Exception as e:
                logger.warning("Ollama failed: %s — falling back to Claude", e)

        response = self.claude.generate(prompt, system)
        return response, f"claude/{self.claude.model}"
