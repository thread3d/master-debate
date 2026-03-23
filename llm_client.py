import time
from typing import Optional
import requests


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-coder-next:q8_0",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = 3000

    def generate(self, prompt: str, system_prompt: str = "") -> Optional[str]:
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

            result = response.json()
            response_text = result.get("response", "").strip()

            # Remove thinking tags and content between them
            import re

            # Remove <think>...</think> tags and everything between them
            response_text = re.sub(
                r"<think>.*?</think>", "", response_text, flags=re.DOTALL
            )
            # Also remove any remaining thinking tags that might not be properly closed
            response_text = re.sub(r"<think>.*", "", response_text, flags=re.DOTALL)
            response_text = response_text.strip()

            return response_text

        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM: {e}")
            return None

    def chat(self, model: str, messages: list) -> Optional[str]:
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

            result = response.json()
            return result.get("message", {}).get("content", "").strip()

        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM: {e}")
            return None
