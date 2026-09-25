import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

import app
from llm_client import LLMClient

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _start_button(at):
    return next(button for button in at.button if "Start Debate" in button.label)


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
    assert loaded["model_count"] == 3


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


def test_display_debate_results_ignores_empty_input():
    app.display_debate_results(None)  # must not raise


# --- end-to-end through Streamlit's AppTest harness -------------------------


def test_app_boots_without_exception(app_env):
    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not at.exception


def test_full_debate_flow(app_env, monkeypatch):
    counter = {"n": 0}

    def fake_generate(self, prompt, system_prompt=""):
        if "Has consensus been reached" in prompt:
            return "NO"
        if "Provide a concise summary" in prompt:
            return "Summary text."
        counter["n"] += 1
        return f"Argument {counter['n']}"

    monkeypatch.setattr(LLMClient, "generate", fake_generate)

    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    at.text_area[0].set_value("Is AI capable of consciousness?").run()
    _start_button(at).click().run()

    assert not at.exception
    results = at.session_state["results"]
    assert results["issue"] == "Is AI capable of consciousness?"
    assert results["result"]["turns"] > 0
    assert results["history"]

    saved = list((app_env / "debates").glob("*.json"))
    assert len(saved) == 1
    persisted = json.loads(saved[0].read_text())
    assert persisted["settings"]["judge_model"]
    assert persisted["settings"]["selected_model"]


def test_starting_without_an_issue_warns_and_resets(app_env, monkeypatch):
    monkeypatch.setattr(LLMClient, "generate", lambda self, prompt, system="": "x")

    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    _start_button(at).click().run()

    assert not at.exception
    assert any("enter a debate issue" in warning.value for warning in at.warning)
    assert at.session_state["running"] is False
    assert at.session_state["results"] is None


def test_history_debate_can_be_loaded_without_error(app_env, monkeypatch):
    monkeypatch.setattr(LLMClient, "generate", lambda self, prompt, system="": "x")

    # Seed one saved debate in the redirected data directory.
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

    history_buttons = [
        button for button in at.button if button.label.startswith("💬")
    ]
    assert history_buttons, "expected the saved debate to appear in the sidebar"
    history_buttons[0].click().run()

    assert not at.exception
    assert at.session_state["results"]["issue"] == "Seeded issue"


# --- error handling and edge cases ------------------------------------------


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


def test_history_with_invalid_timestamp_does_not_crash(app_env, monkeypatch):
    monkeypatch.setattr(LLMClient, "generate", lambda self, prompt, system="": "x")

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


def test_reset_button_clears_session_state(app_env):
    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    at.session_state["_settings"]["selected_model"] = "session-only-model"
    at.run()

    reset = next(button for button in at.button if "Reset" in button.label)
    reset.click().run()

    assert not at.exception
    assert at.session_state["_settings"]["selected_model"] != "session-only-model"


def test_app_enforces_the_time_limit_end_to_end(app_env, monkeypatch):
    """A huge clock jump must stop the debate and show the time-limit banner."""
    import debate_engine

    clock = {"t": 0.0}

    def fast_forward():
        clock["t"] += 1_000_000_000.0
        return clock["t"]

    monkeypatch.setattr(debate_engine.time, "monotonic", fast_forward)
    monkeypatch.setattr(LLMClient, "generate", lambda self, prompt, system="": "NO")

    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    at.text_area[0].set_value("Does time fly?").run()
    _start_button(at).click().run()

    assert not at.exception
    result = at.session_state["results"]["result"]
    assert result["time_limit_reached"] is True
    assert any("Time limit reached" in warning.value for warning in at.warning)
