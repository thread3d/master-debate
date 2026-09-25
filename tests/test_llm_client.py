import json

import pytest
import requests

from llm_client import LLMClient, LLMError, list_models, strip_thinking

TAG = chr(60) + "think" + chr(62)
END = chr(60) + "/think" + chr(62)


class FakeResponse:
    def __init__(self, payload, status_code=200, lines=None, unparsable=False):
        self._payload = payload
        self.status_code = status_code
        self._lines = lines or []
        self._unparsable = unparsable

    def json(self):
        if self._unparsable:
            raise json.JSONDecodeError("bad", "", 0)
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def iter_lines(self):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def ndjson(*chunks):
    return [json.dumps(chunk).encode() for chunk in chunks]


def test_base_url_trailing_slash_is_stripped():
    assert LLMClient("http://localhost:11434/").base_url == "http://localhost:11434"


def test_default_timeout_and_override():
    assert LLMClient().timeout == 3000
    assert LLMClient(timeout=12).timeout == 12


# --- generate ---------------------------------------------------------------


def test_generate_sends_ollama_payload(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None, stream=False):
        captured.update(url=url, payload=json, timeout=timeout, stream=stream)
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


def test_generate_handles_missing_response_field(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse({}))
    assert LLMClient().generate("p") == ""


def test_generate_strips_thinking_blocks(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: FakeResponse({"response": TAG + "secret" + END + " answer "}),
    )
    assert LLMClient().generate("p") == "answer"


# --- error surfacing --------------------------------------------------------


def test_generate_raises_on_transport_error(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(LLMError, match="could not reach"):
        LLMClient().generate("p")


def test_generate_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse({}, 500))
    with pytest.raises(LLMError, match="HTTP 500"):
        LLMClient().generate("p")


def test_http_error_message_names_a_missing_model(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: FakeResponse({"error": "model 'ghost' not found"}, 404),
    )
    with pytest.raises(LLMError, match="model 'ghost' not found"):
        LLMClient().generate("p")


def test_generate_raises_on_malformed_json(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse({}, unparsable=True))
    with pytest.raises(LLMError, match="unexpected response"):
        LLMClient().generate("p")


def test_generate_raises_on_non_object_json(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(["nope"]))
    with pytest.raises(LLMError, match="unexpected response"):
        LLMClient().generate("p")


# --- streaming --------------------------------------------------------------


def test_stream_generate_accumulates_and_cleans(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None, stream=False):
        captured.update(payload=json, stream=stream)
        return FakeResponse(
            {},
            lines=ndjson(
                {"response": TAG + "reasoning" + END},
                {"response": " Answer"},
                {"done": True},
            ),
        )

    monkeypatch.setattr(requests, "post", fake_post)
    pieces = list(LLMClient().stream_generate("p"))

    assert pieces == ["", "Answer"]
    assert captured["payload"]["stream"] is True
    assert captured["stream"] is True


def test_stream_generate_ignores_noise_lines(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: FakeResponse(
            {},
            lines=[
                b"",
                b"not json",
                b"123",
                json.dumps({"response": "x"}).encode(),
                json.dumps({"done": True}).encode(),
            ],
        ),
    )
    assert list(LLMClient().stream_generate("p")) == ["x"]


def test_stream_generate_raises_on_error_chunk(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: FakeResponse({}, lines=ndjson({"error": "model not found"})),
    )
    with pytest.raises(LLMError, match="model not found"):
        list(LLMClient().stream_generate("p"))


def test_stream_generate_raises_on_transport_error(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(LLMError, match="could not reach"):
        list(LLMClient().stream_generate("p"))


def test_stream_generate_raises_on_mid_stream_failure(monkeypatch):
    class Failing(FakeResponse):
        def iter_lines(self):
            raise requests.exceptions.ChunkedEncodingError("broken pipe")

    monkeypatch.setattr(requests, "post", lambda *a, **k: Failing({}))
    with pytest.raises(LLMError, match="stream from"):
        list(LLMClient().stream_generate("p"))


# --- chat -------------------------------------------------------------------


def test_chat_sends_ollama_payload(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None, stream=False):
        captured.update(url=url, payload=json)
        return FakeResponse({"message": {"content": "hi there"}})

    monkeypatch.setattr(requests, "post", fake_post)
    messages = [{"role": "user", "content": "hello"}]

    assert LLMClient().chat("chat-model", messages) == "hi there"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["model"] == "chat-model"
    assert captured["payload"]["messages"] == messages


def test_chat_handles_null_message(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse({"message": None}))
    assert LLMClient().chat("m", []) == ""


def test_chat_raises_on_non_object_json(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse("not a dict"))
    with pytest.raises(LLMError):
        LLMClient().chat("m", [])


# --- model discovery --------------------------------------------------------


def test_list_models_parses_names(monkeypatch):
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: FakeResponse(
            {"models": [{"name": "a:latest"}, {"name": "b"}, {"size": 1}]}
        ),
    )
    assert list_models("http://localhost:11434/") == ["a:latest", "b"]


def test_list_models_returns_empty_when_unreachable(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", boom)
    assert list_models("http://localhost:11434") == []


def test_list_models_returns_empty_for_unexpected_payload(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse({"models": "nope"}))
    assert list_models("http://localhost:11434") == []


# --- thinking-tag cleanup ---------------------------------------------------


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
    assert strip_thinking(text) == expected
