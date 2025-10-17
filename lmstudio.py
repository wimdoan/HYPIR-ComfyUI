from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


class PromptEnhancementError(RuntimeError):
    """Raised when the LM Studio endpoint cannot enhance the prompt."""


@dataclass
class LMStudioConfig:
    endpoint: str
    model: str
    system_prompt: str
    temperature: float
    max_tokens: int
    timeout: float = 30.0


class LMStudioClient:
    def __init__(self, config: LMStudioConfig):
        self.config = config
        self._endpoint = config.endpoint.rstrip("/") + "/v1/chat/completions"

    def enhance_prompt(self, prompt: str) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Take the user's restoration notes below and polish them into a vivid, detailed"
                        " prompt for a Stable Diffusion 2.1 HYPIR LoRA upscaler. Keep every subject,"
                        " composition cue, and explicit instruction the user provides—only refine wording,"
                        " clarify lighting, and describe desired restoration outcomes. Do not ask the user"
                        " for additional information. If the notes are empty, immediately return the short,"
                        " general restoration prompt provided in your system instructions."
                        "\n\nUser notes:\n" + prompt
                    ),
                },
            ],
            "temperature": float(self.config.temperature),
            "max_tokens": int(self.config.max_tokens),
        }

        try:
            response = requests.post(
                self._endpoint,
                json=payload,
                timeout=self.config.timeout,
            )
        except requests.RequestException as exc:
            raise PromptEnhancementError(f"Failed to reach LM Studio endpoint: {exc}") from exc

        if response.status_code != 200:
            raise PromptEnhancementError(
                f"LM Studio returned status {response.status_code}: {response.text[:200]}"
            )

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise PromptEnhancementError("LM Studio response did not include any choices.")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise PromptEnhancementError("LM Studio response did not include message content.")

        logger.debug("LM Studio enhanced prompt: %s", content)
        return content.strip()
