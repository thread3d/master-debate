import re

import requests


def _strip_thinking(text: str) -> str:
    """Strip model reasoning blocks from a reply.

    Handles complete blocks, an unclosed opening tag, and the orphan
    closing tag some servers emit after stripping the opening tag.
    """
    # Complete <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # A stray closing tag with no opener: everything before it was reasoning
    text = re.sub(r"^.*</think>", "", text, flags=re.DOTALL)
    # An unclosed opening tag: drop everything from it onward
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-coder-next:q8_0",
        timeout: float = 3000,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @staticmethod
    def _decode(response) -> dict | None:
        """Parse a JSON object from a response body.

        Returns None for bodies that are not a JSON object, so callers can treat
        malformed responses the same way they treat transport failures.
        """
        try:
            result = response.json()
        except ValueError:
            return None
        return result if isinstance(result, dict) else None

    def generate(self, prompt: str, system_prompt: str = "") -> str | None:
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 512},
            }

            response = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
            )
            response.raise_for_status()

            result = self._decode(response)
            if result is None:
                return None
            return _strip_thinking(result.get("response") or "")

        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM: {e}")
            return None

    def chat(self, model: str, messages: list) -> str | None:
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 512},
            }

            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
            )
            response.raise_for_status()

            result = self._decode(response)
            if result is None:
                return None
            message = result.get("message") or {}
            content = message.get("content") if isinstance(message, dict) else ""
            return _strip_thinking(content or "")

        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM: {e}")
            return None
