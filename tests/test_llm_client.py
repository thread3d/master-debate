import json

import pytest
import requests

from llm_client import LLMClient, _strip_thinking

TAG = chr(60) + "think" + chr(62)
END = chr(60) + "/think" + chr(62)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


class BadJSONResponse:
    def raise_for_status(self):
        pass

    def json(self):
        raise json.JSONDecodeError("bad", "", 0)


def test_base_url_trailing_slash_is_stripped():
    assert LLMClient("http://localhost:11434/").base_url == "http://localhost:11434"


def test_default_timeout_and_override():
    assert LLMClient().timeout == 3000
    assert LLMClient(timeout=12).timeout == 12


def test_generate_sends_ollama_payload(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(url=url, payload=json, timeout=timeout)
        return FakeResponse({"response": "hello"})

    monkeypatch.setattr(requests, "post", fake_post)
    client = LLMClient("http://localhost:11434", "my-model", timeout=42)

    assert client.generate("the prompt", "the system") == "hello"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["payload"]["model"] == "my-model"
    assert captured["payload"]["prompt"] == "the prompt"
    assert captured["payload"]["system"] == "the system"
    assert captured["payload"]["stream"] is False
    assert captured["timeout"] == 42


def test_chat_sends_ollama_payload(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(url=url, payload=json)
        return FakeResponse({"message": {"content": "hi there"}})

    monkeypatch.setattr(requests, "post", fake_post)
    client = LLMClient("http://localhost:11434")
    messages = [{"role": "user", "content": "hello"}]

    assert client.chat("chat-model", messages) == "hi there"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["model"] == "chat-model"
    assert captured["payload"]["messages"] == messages


def test_generate_strips_thinking_blocks(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: FakeResponse({"response": TAG + "secret" + END + " answer "}),
    )
    assert LLMClient().generate("p") == "answer"


def test_chat_strips_thinking_blocks(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: FakeResponse(
            {"message": {"content": "answer " + TAG + "unclosed reasoning"}}
        ),
    )
    assert LLMClient().chat("m", []) == "answer"


def test_generate_handles_missing_response_field(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse({}))
    assert LLMClient().generate("p") == ""


def test_chat_handles_null_message(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse({"message": None}))
    assert LLMClient().chat("m", []) == ""


def test_transport_error_returns_none(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", boom)
    client = LLMClient()
    assert client.generate("p") is None
    assert client.chat("m", []) is None


def test_http_error_returns_none(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse({}, 500))
    assert LLMClient().generate("p") is None


def test_malformed_json_returns_none(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: BadJSONResponse())
    assert LLMClient().generate("p") is None


def test_non_object_json_returns_none(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(["nope"]))
    assert LLMClient().generate("p") is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("plain answer", "plain answer"),
        (TAG + "hidden" + END + "visible", "visible"),
        ("before " + TAG + "unclosed", "before"),
        (TAG + "a" + END + "middle" + TAG + "b" + END, "middle"),
        ("reasoning" + END + "answer", "answer"),
        ("a" + END + "b" + END + "final", "final"),
        ("The user wants one word.\n" + END + "\n\nOK", "OK"),
        ("  spaced  ", "spaced"),
    ],
)
def test_strip_thinking_cases(text, expected):
    assert _strip_thinking(text) == expected


def test_chat_non_object_json_returns_none(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse("not a dict"))
    assert LLMClient().chat("m", []) is None


def test_generate_non_object_json_returns_none(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(42))
    assert LLMClient().generate("p") is None
