import json
import re

import requests


class LLMError(RuntimeError):
    """Raised when a call to the LLM backend fails."""


def list_models(base_url: str, timeout: float = 3) -> list[str]:
    """Return model names advertised by an Ollama-compatible server.

    Returns an empty list when the server is unreachable or the payload is
    unexpected; callers fall back to their own built-in list.
    """
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except (requests.exceptions.RequestException, ValueError):
        return []

    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return []

    names = []
    for entry in models:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            names.append(entry["name"])
    return names


def strip_thinking(text: str) -> str:
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

        Returns None for bodies that are not a JSON object.
        """
        try:
            result = response.json()
        except ValueError:
            return None
        return result if isinstance(result, dict) else None

    def _post(self, path: str, payload: dict, stream: bool = False):
        """POST to the backend, translating transport failures into LLMError."""
        try:
            response = requests.post(
                f"{self.base_url}{path}",
                json=payload,
                timeout=self.timeout,
                stream=stream,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise LLMError(self._http_error_message(response)) from e
        except requests.exceptions.RequestException as e:
            raise LLMError(f"could not reach {self.base_url}: {e}") from e
        return response

    def _http_error_message(self, response) -> str:
        """Turn an HTTP error into a message that names the actual problem.

        Ollama reports a missing model as HTTP 404 with an ``error`` field, so
        blindly calling that "could not reach the server" would be misleading.
        """
        body = self._decode(response)
        detail = f": {body['error']}" if body and body.get("error") else ""
        return f"{self.base_url} returned HTTP {response.status_code}{detail}"

    def _payload(self, prompt: str, system_prompt: str, stream: bool) -> dict:
        return {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": stream,
            "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 512},
        }

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Return a complete response, or raise LLMError on failure."""
        response = self._post(
            "/api/generate", self._payload(prompt, system_prompt, stream=False)
        )
        result = self._decode(response)
        if result is None:
            raise LLMError(f"{self.base_url} returned an unexpected response")
        return strip_thinking(result.get("response") or "")

    def stream_generate(self, prompt: str, system_prompt: str = ""):
        """Yield cleaned, accumulated response text as the model produces it.

        Each yielded value is the whole reply cleaned so far, so callers can
        render it directly. Raises LLMError on a failed request or stream.
        """
        buffer = ""
        payload = self._payload(prompt, system_prompt, stream=True)
        with self._post("/api/generate", payload, stream=True) as response:
            try:
                for piece in self._iter_chunks(response):
                    buffer += piece
                    yield strip_thinking(buffer)
            except requests.exceptions.RequestException as e:
                raise LLMError(f"stream from {self.base_url} failed: {e}") from e

    @staticmethod
    def _iter_chunks(response):
        """Yield raw text pieces from an Ollama NDJSON response stream."""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except ValueError:
                continue
            if not isinstance(chunk, dict):
                continue
            if chunk.get("error"):
                raise LLMError(str(chunk["error"]))
            if chunk.get("response"):
                yield chunk["response"]
            if chunk.get("done"):
                return

    def chat(self, model: str, messages: list) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 512},
        }
        response = self._post("/api/chat", payload)
        result = self._decode(response)
        if result is None:
            raise LLMError(f"{self.base_url} returned an unexpected response")
        message = result.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else ""
        return strip_thinking(content or "")
