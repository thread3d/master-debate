import debate_engine
from debate_engine import DebateStopped, PhilosopherDebate
from llm_client import LLMError


class FakeClient:
    """Minimal stand-in for LLMClient that records prompts."""

    def __init__(self, replies=None, default="response"):
        self.replies = list(replies or [])
        self.default = default
        self.calls = []

    def generate(self, prompt, system_prompt=""):
        self.calls.append((prompt, system_prompt))
        if self.replies:
            return self.replies.pop(0)
        return self.default


def prompts(client):
    return [prompt for prompt, _ in client.calls]


# --- client selection -------------------------------------------------------


def test_single_client_is_shared_by_all_philosophers():
    client = FakeClient()
    debate = PhilosopherDebate(["Socrates", "Plato"], "issue", client)
    assert debate.generate_debate_turn(0) == "response"
    assert debate.generate_debate_turn(1) == "response"
    assert len(client.calls) == 2


def test_mapping_selects_the_matching_client():
    socrates = FakeClient(default="socratic")
    plato = FakeClient(default="platonic")
    debate = PhilosopherDebate(
        ["Socrates", "Plato"], "issue", {"Socrates": socrates, "Plato": plato}
    )
    assert debate.generate_debate_turn(0) == "socratic"
    assert debate.generate_debate_turn(1) == "platonic"


def test_mapping_falls_back_to_first_client():
    only = FakeClient(default="fallback")
    debate = PhilosopherDebate(["Socrates", "Unknown"], "issue", {"Socrates": only})
    assert debate.generate_debate_turn(1) == "fallback"


def test_empty_mapping_is_handled_without_crashing():
    debate = PhilosopherDebate(["Socrates"], "issue", {})
    assert debate.generate_debate_turn(0) == ""
    assert debate.check_consensus() is False
    assert debate.get_summary() == "No debate occurred."


def test_unknown_philosopher_uses_empty_persona_but_still_calls_client():
    client = FakeClient(default="still works")
    debate = PhilosopherDebate(["Nobody"], "issue", client)
    assert debate.generate_debate_turn(0) == "still works"
    assert "Nobody" in prompts(client)[0]


# --- prompts and context ----------------------------------------------------


def test_prompt_includes_persona_and_issue():
    client = FakeClient()
    debate = PhilosopherDebate(["Socrates"], "Is knowledge possible?", client)
    debate.generate_debate_turn(0)
    prompt, system = client.calls[0]
    assert "Is knowledge possible?" in prompt
    assert "Socrates" in prompt
    assert system  # persona system prompt is passed through


def test_context_is_capped_at_six_entries():
    debate = PhilosopherDebate(["Socrates"], "issue", FakeClient())
    debate.history = [
        {"turn": i, "name": f"Phil{i}", "text": f"text{i}"} for i in range(8)
    ]
    context = debate._build_context(8)
    assert "Phil2" in context
    assert "Phil1" not in context


def test_first_turn_context_has_no_history():
    debate = PhilosopherDebate(["Socrates"], "issue", FakeClient())
    assert debate._build_context(0).startswith("Initial issue: issue")


def test_philosophers_rotate_in_order():
    client = FakeClient(default="x")
    debate = PhilosopherDebate(["A", "B", "C"], "issue", client, max_turns=6)
    debate.run_debate()
    assert [entry["name"] for entry in debate.history] == ["A", "B", "C", "A", "B", "C"]


# --- consensus and judging --------------------------------------------------


def test_consensus_short_circuits_the_debate():
    client = FakeClient(replies=["first", "second", "YES"], default="NO")
    debate = PhilosopherDebate(["A", "B"], "issue", client)
    result = debate.run_debate()
    assert result["consensus_reached"] is True
    assert result["turns"] == 2


def test_consensus_is_not_checked_before_two_turns():
    client = FakeClient(default="NO")
    debate = PhilosopherDebate(["A", "B"], "issue", client, max_turns=2)
    debate.run_debate()
    assert sum("Has consensus been reached" in p for p in prompts(client)) == 1


def test_dedicated_judge_client_handles_consensus_and_summary():
    philosopher = FakeClient(default="argument")
    judge = FakeClient(default="YES")
    debate = PhilosopherDebate(
        ["A", "B"], "issue", philosopher, judge_client=judge
    )
    result = debate.run_debate()
    assert result["consensus_reached"] is True

    # The debaters never see the judging prompts...
    assert not any("Has consensus been reached" in p for p in prompts(philosopher))
    # ...the dedicated judge does.
    assert any("Has consensus been reached" in p for p in prompts(judge))

    debate.history = [{"turn": 0, "name": "A", "text": "hello"}]
    assert debate.get_summary() == "YES"
    assert any("Provide a concise summary" in p for p in prompts(judge))


def test_judge_falls_back_to_a_philosopher_when_not_injected():
    client = FakeClient(default="NO")
    debate = PhilosopherDebate(["A", "B"], "issue", client)
    debate.run_debate()
    assert any("Has consensus been reached" in p for p in prompts(client))


# --- termination paths ------------------------------------------------------


def test_max_turns_termination():
    client = FakeClient(default="NO")
    debate = PhilosopherDebate(["A", "B"], "issue", client, max_turns=4)
    result = debate.run_debate()
    assert result == {
        "consensus_reached": False,
        "turns": 4,
        "max_turns_reached": True,
    }
    assert len(debate.history) == 4


def test_run_debate_without_philosophers_returns_error():
    result = PhilosopherDebate([], "issue", FakeClient()).run_debate()
    assert result["turns"] == 0
    assert "error" in result


def test_zero_time_limit_stops_immediately():
    debate = PhilosopherDebate(
        ["A"], "issue", FakeClient(), time_limit_seconds=0
    )
    result = debate.run_debate()
    assert result["time_limit_reached"] is True
    assert result["turns"] == 0


def test_time_limit_stops_a_running_debate(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(debate_engine.time, "monotonic", lambda: clock["t"])

    class SlowClient(FakeClient):
        def generate(self, prompt, system_prompt=""):
            clock["t"] += 10
            return super().generate(prompt, system_prompt)

    debate = PhilosopherDebate(
        ["A", "B"],
        "issue",
        SlowClient(default="NO"),
        max_turns=20,
        time_limit_seconds=25,
    )
    result = debate.run_debate()
    assert result["time_limit_reached"] is True
    assert 1 <= result["turns"] < 20


def test_no_time_limit_runs_to_completion():
    debate = PhilosopherDebate(
        ["A", "B"], "issue", FakeClient(default="NO"), max_turns=3
    )
    result = debate.run_debate()
    assert result["max_turns_reached"] is True
    assert "time_limit_reached" not in result


# --- callbacks and summary --------------------------------------------------


def test_on_turn_callback_receives_each_turn():
    seen = []
    debate = PhilosopherDebate(["A", "B"], "issue", FakeClient(default="NO"), max_turns=3)
    debate.run_debate(on_turn_callback=lambda n, name, text: seen.append((n, name, text)))
    assert [n for n, _, _ in seen] == [0, 1, 2]
    assert [name for _, name, _ in seen] == ["A", "B", "A"]


def test_empty_responses_are_not_recorded():
    client = FakeClient(replies=["", "real answer"])
    debate = PhilosopherDebate(["A", "B"], "issue", client, max_turns=2)
    result = debate.run_debate()
    assert result["turns"] == 1
    assert debate.history[0]["text"] == "real answer"


def test_summary_falls_back_when_client_returns_nothing():
    client = FakeClient(default="")
    debate = PhilosopherDebate(["A"], "issue", client)
    debate.history = [{"turn": 0, "name": "A", "text": "hello"}]
    assert debate.get_summary() == "Debate completed."


def test_judge_client_falls_back_when_no_philosophers_configured():
    client = FakeClient(default="x")
    debate = PhilosopherDebate([], "issue", client)
    assert debate._get_judge_client() is client


def test_generate_turn_without_philosophers_returns_empty():
    debate = PhilosopherDebate([], "issue", FakeClient())
    assert debate.generate_debate_turn(0) == ""


def test_consensus_returns_false_when_no_client_available():
    debate = PhilosopherDebate(["A"], "issue", {})
    debate.history = [{"turn": 0, "name": "A", "text": "hi"}]
    assert debate.check_consensus() is False


def test_consensus_returns_false_for_empty_judge_reply():
    debate = PhilosopherDebate(["A"], "issue", FakeClient(default=""))
    debate.history = [{"turn": 0, "name": "A", "text": "hi"}]
    assert debate.check_consensus() is False


def test_format_full_history_without_history_uses_issue():
    debate = PhilosopherDebate(["A"], "the issue", FakeClient())
    assert debate._format_full_history() == "Initial issue: the issue"


def test_summary_returns_fallback_when_no_client_available():
    debate = PhilosopherDebate(["A"], "issue", {})
    debate.history = [{"turn": 0, "name": "A", "text": "hi"}]
    assert debate.get_summary() == "Debate completed."


# --- backend failures -------------------------------------------------------


class DeadClient:
    def generate(self, prompt, system_prompt=""):
        raise LLMError("could not reach http://localhost:11434")


class FlakyClient:
    def __init__(self, failures=1, text="recovered"):
        self.failures = failures
        self.text = text
        self.calls = 0

    def generate(self, prompt, system_prompt=""):
        self.calls += 1
        if self.calls <= self.failures:
            raise LLMError("backend down")
        return self.text


class BrokenJudge:
    def generate(self, prompt, system_prompt=""):
        raise LLMError("judge offline")


class StreamingClient:
    def __init__(self, pieces):
        self.pieces = list(pieces)

    def generate(self, prompt, system_prompt=""):
        return "".join(self.pieces)

    def stream_generate(self, prompt, system_prompt=""):
        buffer = ""
        for piece in self.pieces:
            buffer += piece
            yield buffer


def test_backend_failure_aborts_and_reports_the_error():
    debate = PhilosopherDebate(["A", "B"], "issue", DeadClient())
    result = debate.run_debate()
    assert result["turns"] == 0
    assert "could not reach" in result["error"]
    assert result["failed_turns"] == 2
    assert len(result["errors"]) == 2


def test_transient_failure_is_tolerated_and_recorded():
    # The first attempt fails and consumes a turn slot, so 3 max turns yields 2
    # recorded turns plus one failure.
    debate = PhilosopherDebate(["A", "B"], "issue", FlakyClient(failures=1), max_turns=3)
    result = debate.run_debate()
    assert result["turns"] == 2
    assert result["failed_turns"] == 1


def test_judge_failure_does_not_end_the_debate():
    debate = PhilosopherDebate(
        ["A", "B"],
        "issue",
        FakeClient(default="argument"),
        judge_client=BrokenJudge(),
        max_turns=3,
    )
    result = debate.run_debate()
    assert result["turns"] == 3
    # Consensus is re-checked every turn from the second onwards.
    assert result["judge_errors"] == ["judge offline", "judge offline"]
    assert result["consensus_reached"] is False


def test_summary_reports_judge_failure():
    debate = PhilosopherDebate(["A"], "issue", {}, judge_client=BrokenJudge())
    debate.history = [{"turn": 0, "name": "A", "text": "hi"}]
    assert debate.get_summary().startswith("Summary unavailable")


# --- streaming and stopping -------------------------------------------------


def test_streaming_callback_receives_partial_text():
    debate = PhilosopherDebate(
        ["A"], "issue", StreamingClient(["Hel", "lo ", "world"]), max_turns=1
    )
    seen = []
    debate.run_debate(
        on_chunk_callback=lambda name, text: seen.append((name, text))
    )
    assert seen == [("A", "Hel"), ("A", "Hello "), ("A", "Hello world")]
    assert debate.history[0]["text"] == "Hello world"


def test_stop_before_the_first_turn():
    debate = PhilosopherDebate(["A"], "issue", FakeClient(), max_turns=5)
    result = debate.run_debate(should_stop=lambda: True)
    assert result["stopped"] is True
    assert result["turns"] == 0


def test_stop_request_ends_the_debate_after_a_turn():
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 2

    debate = PhilosopherDebate(["A", "B"], "issue", FakeClient(default="NO"), max_turns=20)
    result = debate.run_debate(should_stop=should_stop)
    assert result["stopped"] is True
    assert result["turns"] == 1


def test_chunk_callback_can_abort_with_debate_stopped():
    def abort(name, text):
        raise DebateStopped

    debate = PhilosopherDebate(["A"], "issue", StreamingClient(["a", "b"]), max_turns=5)
    result = debate.run_debate(on_chunk_callback=abort)
    assert result["stopped"] is True
    assert result["turns"] == 0
