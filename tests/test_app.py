import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

import app
import llm_client

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _canned_reply(prompt):
    if "Has consensus been reached" in prompt:
        return "NO"
    if "Provide a concise summary" in prompt:
        return "SUMMARY-MARKER"
    return "argument"


def _start_button(at):
    return next(button for button in at.button if "Start Debate" in button.label)


@pytest.fixture
def no_model_discovery(monkeypatch):
    """Keep AppTest runs hermetic and offline."""
    monkeypatch.setattr(llm_client, "list_models", lambda base_url, timeout=3: [])


@pytest.fixture
def fake_llm(monkeypatch):
    def generate(self, prompt, system_prompt=""):
        return _canned_reply(prompt)

    def stream_generate(self, prompt, system_prompt=""):
        yield _canned_reply(prompt)

    monkeypatch.setattr(llm_client.LLMClient, "generate", generate)
    monkeypatch.setattr(llm_client.LLMClient, "stream_generate", stream_generate)


# --- pure helpers -----------------------------------------------------------


@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "00:00"), (59.9, "00:59"), (61, "01:01"), (3600, "60:00")],
)
def test_format_time(seconds, expected):
    assert app.format_time(seconds) == expected


@pytest.mark.parametrize(
    "result,level",
    [
        ({"consensus_reached": True}, "success"),
        ({"consensus_reached": False, "stopped": True}, "warning"),
        ({"consensus_reached": False, "time_limit_reached": True}, "warning"),
        ({"consensus_reached": False, "max_turns_reached": True}, "warning"),
        ({"consensus_reached": False, "error": "boom"}, "error"),
        ({"consensus_reached": False}, "warning"),
    ],
)
def test_debate_outcome_levels(result, level):
    got_level, message = app.debate_outcome(result)
    assert got_level == level
    assert message


# --- settings and storage ---------------------------------------------------


def test_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    defaults = app.load_settings()
    assert defaults["time_limit"] == app.DEFAULT_TIME_LIMIT_MINUTES
    assert defaults["judge_model"]
    assert defaults["per_philosopher_overrides"] is False

    defaults["selected_model"] = "custom-model"
    app.save_settings(defaults)
    assert app.load_settings()["selected_model"] == "custom-model"


def test_load_settings_merges_missing_keys(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({"base_url": "http://example"}))
    monkeypatch.setattr(app, "SETTINGS_FILE", str(settings_file))

    loaded = app.load_settings()
    assert loaded["base_url"] == "http://example"
    assert loaded["judge_model"]
    assert "per_philosopher_overrides" in loaded


def test_debate_storage_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DEBATES_DIR", str(tmp_path / "debates"))
    assert app.load_all_debates() == []

    app.save_debate({"issue": "first", "timestamp": "2025-01-01T00:00:00"})
    app.save_debate({"issue": "second", "timestamp": "2025-01-02T00:00:00"})

    assert [d["issue"] for d in app.load_all_debates()] == ["second", "first"]


def test_save_debate_creates_missing_directory(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "debates"
    monkeypatch.setattr(app, "DEBATES_DIR", str(target))
    app.save_debate({"issue": "x", "timestamp": "2025-01-01T00:00:00"})
    assert target.exists()


# --- results rendering ------------------------------------------------------


def test_display_debate_results_supports_nested_and_legacy(monkeypatch):
    fake_st = MagicMock()
    fake_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
    monkeypatch.setattr(app, "st", fake_st)

    nested = {
        "issue": "Is AI conscious?",
        "philosophers": ["Socrates", "Plato"],
        "result": {"consensus_reached": False, "turns": 4, "max_turns_reached": True},
        "history": [{"turn": 0, "name": "Socrates", "text": "hi"}],
        "summary": "A summary.",
        "timestamp": "2025-01-01T00:00:00",
        "settings": {
            "base_url": "http://localhost:11434",
            "selected_model": "m",
            "model_count": 2,
            "time_limit": 5,
        },
    }
    app.display_debate_results(nested)

    legacy = {k: v for k, v in nested.items() if k != "settings"}
    legacy.update(base_url="http://x", selected_model="m", model_count=2, time_limit=5)
    app.display_debate_results(legacy)

    assert fake_st.caption.called
    assert fake_st.warning.called


def test_display_debate_results_error_outcome(monkeypatch):
    fake_st = MagicMock()
    fake_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
    monkeypatch.setattr(app, "st", fake_st)

    app.display_debate_results(
        {
            "issue": "x",
            "philosophers": [],
            "result": {"consensus_reached": False, "error": "no philosophers"},
            "history": [],
            "settings": {},
        }
    )
    assert fake_st.error.called


def test_display_debate_results_ignores_empty_input():
    app.display_debate_results(None)  # must not raise


# --- client construction ----------------------------------------------------


def test_build_clients_ignores_overrides_when_disabled():
    settings = {
        "base_url": "http://global",
        "selected_model": "global-model",
        "judge_model": "judge-model",
        "per_philosopher_overrides": False,
    }
    overrides = {"Socrates": {"base_url": "http://local", "model": "local-model"}}

    clients, judge = app.build_clients(["Socrates"], overrides, settings)

    assert clients["Socrates"].base_url == "http://global"
    assert clients["Socrates"].model == "global-model"
    assert judge.model == "judge-model"


def test_build_clients_applies_overrides_when_enabled():
    settings = {
        "base_url": "http://global",
        "selected_model": "global-model",
        "judge_model": "judge-model",
        "per_philosopher_overrides": True,
    }
    overrides = {"Socrates": {"base_url": "http://local", "model": "local-model"}}

    clients, _ = app.build_clients(["Socrates", "Plato"], overrides, settings)

    assert clients["Socrates"].base_url == "http://local"
    assert clients["Socrates"].model == "local-model"
    # Philosophers without an override fall back to the global settings.
    assert clients["Plato"].base_url == "http://global"
    assert clients["Plato"].model == "global-model"


# --- error handling ---------------------------------------------------------


def test_load_settings_recovers_from_corrupt_file(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    settings_file.write_text("{ not valid json")
    monkeypatch.setattr(app, "SETTINGS_FILE", str(settings_file))
    fake_st = MagicMock()
    monkeypatch.setattr(app, "st", fake_st)

    loaded = app.load_settings()
    assert loaded["time_limit"] == app.DEFAULT_TIME_LIMIT_MINUTES
    assert fake_st.error.called


def test_load_all_debates_skips_corrupt_files(tmp_path, monkeypatch):
    debates_dir = tmp_path / "debates"
    debates_dir.mkdir()
    (debates_dir / "good.json").write_text(
        json.dumps({"issue": "ok", "timestamp": "2025-01-01T00:00:00"})
    )
    (debates_dir / "bad.json").write_text("{ not valid json")

    monkeypatch.setattr(app, "DEBATES_DIR", str(debates_dir))
    fake_st = MagicMock()
    monkeypatch.setattr(app, "st", fake_st)

    loaded = app.load_all_debates()
    assert [d["issue"] for d in loaded] == ["ok"]
    assert fake_st.error.called


def test_save_settings_reports_io_error(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    monkeypatch.setattr(app, "SETTINGS_FILE", str(blocker / "settings.json"))
    fake_st = MagicMock()
    monkeypatch.setattr(app, "st", fake_st)

    app.save_settings({"base_url": "x"})
    assert fake_st.error.called


def test_save_debate_reports_io_error(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    monkeypatch.setattr(app, "DEBATES_DIR", str(blocker / "debates"))
    fake_st = MagicMock()
    monkeypatch.setattr(app, "st", fake_st)

    app.save_debate({"issue": "x", "timestamp": "2025-01-01T00:00:00"})
    assert fake_st.error.called


# --- end-to-end through Streamlit's AppTest harness -------------------------


def test_app_boots_without_exception(app_env, no_model_discovery):
    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not at.exception


def test_full_debate_flow(app_env, no_model_discovery, fake_llm):
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    at.text_area[0].set_value("Is AI capable of consciousness?").run()
    _start_button(at).click().run()

    assert not at.exception
    results = at.session_state["results"]
    assert results["issue"] == "Is AI capable of consciousness?"
    assert results["result"]["turns"] > 0
    assert results["history"]
    assert results["summary"] == "SUMMARY-MARKER"
    assert results["elapsed_seconds"] >= 0
    assert at.session_state.get("_runner") is None

    # The summary is rendered, not just stored.
    assert any("SUMMARY-MARKER" in markdown.value for markdown in at.markdown)

    saved = list((app_env / "debates").glob("*.json"))
    assert len(saved) == 1
    persisted = json.loads(saved[0].read_text())
    assert persisted["summary"] == "SUMMARY-MARKER"
    assert persisted["settings"]["judge_model"]
    assert persisted["settings"]["selected_model"]


def test_backend_failure_is_surfaced_in_the_ui(app_env, no_model_discovery, monkeypatch):
    def boom(self, prompt, system_prompt=""):
        raise llm_client.LLMError("could not reach http://localhost:11434: refused")

    def boom_stream(self, prompt, system_prompt=""):
        raise llm_client.LLMError("could not reach http://localhost:11434: refused")
        yield  # pragma: no cover  (makes this a generator)

    monkeypatch.setattr(llm_client.LLMClient, "generate", boom)
    monkeypatch.setattr(llm_client.LLMClient, "stream_generate", boom_stream)

    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    at.text_area[0].set_value("Unreachable backend?").run()
    _start_button(at).click().run()

    assert not at.exception
    result = at.session_state["results"]["result"]
    assert "could not reach" in result["error"]
    assert any("could not reach" in error.value for error in at.error)


def test_app_enforces_the_time_limit_end_to_end(app_env, no_model_discovery, monkeypatch):
    import debate_engine

    clock = {"t": 0.0}

    def fast_forward():
        clock["t"] += 1_000_000_000.0
        return clock["t"]

    monkeypatch.setattr(debate_engine.time, "monotonic", fast_forward)
    monkeypatch.setattr(
        llm_client.LLMClient, "generate", lambda self, prompt, system="": "NO"
    )

    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    at.text_area[0].set_value("Does time fly?").run()
    _start_button(at).click().run()

    assert not at.exception
    result = at.session_state["results"]["result"]
    assert result["time_limit_reached"] is True
    assert any("Time limit reached" in warning.value for warning in at.warning)


def test_starting_without_an_issue_warns_and_resets(app_env, no_model_discovery):
    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    _start_button(at).click().run()

    assert not at.exception
    assert any("enter a debate issue" in warning.value for warning in at.warning)
    assert at.session_state["running"] is False
    assert at.session_state["results"] is None


def test_discovered_models_appear_in_the_picker(app_env, monkeypatch):
    monkeypatch.setattr(
        llm_client, "list_models", lambda base_url, timeout=3: ["local-only:latest"]
    )
    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not at.exception
    model_box = next(box for box in at.selectbox if box.label == "LLM Model")
    assert "local-only:latest" in model_box.options
    assert app.CUSTOM_OPTION in model_box.options


def test_per_philosopher_overrides_are_hidden_by_default(app_env, no_model_discovery):
    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not any("Socrates" in expander.label for expander in at.expander)

    at.checkbox[0].set_value(True).run()
    assert not at.exception
    assert any("Socrates" in expander.label for expander in at.expander)


def test_custom_model_name_is_saved(app_env, no_model_discovery):
    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    model_box = next(box for box in at.selectbox if box.label == "LLM Model")
    model_box.set_value(app.CUSTOM_OPTION).run()
    assert not at.exception

    custom = next(field for field in at.text_input if "custom name" in field.label)
    custom.set_value("my-own:model").run()

    assert not at.exception
    assert at.session_state["_settings"]["selected_model"] == "my-own:model"


def test_custom_provider_url_is_saved(app_env, no_model_discovery):
    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    provider = next(box for box in at.selectbox if box.label == "LLM Provider URL")
    provider.set_value(app.CUSTOM_OPTION).run()
    assert not at.exception

    field = next(f for f in at.text_input if "custom URL" in f.label)
    field.set_value("http://example.test:1234").run()

    assert not at.exception
    assert at.session_state["_settings"]["base_url"] == "http://example.test:1234"


def test_refresh_model_list_refetches(app_env, monkeypatch):
    calls = {"n": 0}

    def fake_list(base_url, timeout=3):
        calls["n"] += 1
        return ["m1"] if calls["n"] == 1 else ["m2"]

    monkeypatch.setattr(llm_client, "list_models", fake_list)
    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert "m1" in next(box for box in at.selectbox if box.label == "LLM Model").options

    refresh = next(button for button in at.button if "Refresh model list" in button.label)
    refresh.click().run()

    assert not at.exception
    assert "m2" in next(box for box in at.selectbox if box.label == "LLM Model").options


def test_history_debate_can_be_loaded_without_error(app_env, no_model_discovery):
    debates_dir = app_env / "debates"
    debates_dir.mkdir(parents=True, exist_ok=True)
    (debates_dir / "debate_seed.json").write_text(
        json.dumps(
            {
                "issue": "Seeded issue",
                "philosophers": ["Socrates", "Plato"],
                "result": {"consensus_reached": True, "turns": 2},
                "history": [{"turn": 0, "name": "Socrates", "text": "hello"}],
                "timestamp": "2025-01-01T00:00:00",
                "settings": {
                    "base_url": "http://localhost:11434",
                    "selected_model": "m",
                    "model_count": 2,
                    "time_limit": 5,
                },
            }
        )
    )

    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not at.exception

    history_buttons = [button for button in at.button if button.label.startswith("💬")]
    assert history_buttons, "expected the saved debate to appear in the sidebar"
    history_buttons[0].click().run()

    assert not at.exception
    assert at.session_state["results"]["issue"] == "Seeded issue"


def test_history_with_invalid_timestamp_does_not_crash(app_env, no_model_discovery):
    debates_dir = app_env / "debates"
    debates_dir.mkdir(parents=True, exist_ok=True)
    (debates_dir / "debate_bad_ts.json").write_text(
        json.dumps(
            {
                "issue": "Bad timestamp debate",
                "philosophers": ["Socrates", "Plato"],
                "result": {"consensus_reached": False, "turns": 1},
                "history": [],
                "timestamp": "not-a-timestamp",
                "settings": {},
            }
        )
    )

    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not at.exception


def test_reset_button_clears_session_state(app_env, no_model_discovery):
    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    at.session_state["_settings"]["selected_model"] = "session-only-model"
    at.run()

    reset = next(button for button in at.button if "Reset" in button.label)
    reset.click().run()

    assert not at.exception
    assert at.session_state["_settings"]["selected_model"] != "session-only-model"


# --- debate runner integration ----------------------------------------------


def test_runner_state_is_cleared_after_completion(app_env, no_model_discovery, fake_llm):
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    at.text_area[0].set_value("Is time real?").run()
    _start_button(at).click().run()

    assert at.session_state.get("_runner") is None
    assert at.session_state.running is False
