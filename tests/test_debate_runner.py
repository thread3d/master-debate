import time

from debate_engine import PhilosopherDebate
from debate_runner import DebateRunner


class SlowStreamingClient:
    """Streams pieces with a delay so streaming and stopping can be observed."""

    def __init__(self, pieces, delay=0.01):
        self.pieces = list(pieces)
        self.delay = delay

    def generate(self, prompt, system_prompt=""):
        return "".join(self.pieces)

    def stream_generate(self, prompt, system_prompt=""):
        buffer = ""
        for piece in self.pieces:
            time.sleep(self.delay)
            buffer += piece
            yield buffer


def _wait(runner, timeout=5.0):
    deadline = time.monotonic() + timeout
    while runner.is_running() and time.monotonic() < deadline:
        time.sleep(0.01)
    return not runner.is_running()


def test_runner_completes_and_reports_history():
    client = SlowStreamingClient(["a", "b", "c"])
    runner = DebateRunner(PhilosopherDebate(["A"], "issue", client, max_turns=1))
    runner.start()

    assert _wait(runner)
    snapshot = runner.snapshot()
    assert snapshot["result"]["turns"] == 1
    assert snapshot["history"][0]["text"] == "abc"
    assert snapshot["streaming"] is None
    assert snapshot["error"] is None


def test_runner_exposes_partial_text_while_running():
    client = SlowStreamingClient(["one ", "two ", "three"], delay=0.05)
    runner = DebateRunner(PhilosopherDebate(["A"], "issue", client, max_turns=1))
    runner.start()

    seen = []
    deadline = time.monotonic() + 3
    while runner.is_running() and time.monotonic() < deadline:
        partial = runner.snapshot()["streaming"]
        if partial:
            seen.append(partial[1])
        time.sleep(0.01)

    assert _wait(runner)
    assert seen, "expected at least one streamed partial update"
    # Streaming snapshots are inherently racy: the final piece can be replaced
    # by the completed turn before the poll observes it.
    assert all("one two three".startswith(partial) for partial in seen)
    assert runner.result["turns"] == 1


def test_runner_stop_ends_the_debate_early():
    client = SlowStreamingClient(["x"] * 200, delay=0.01)
    runner = DebateRunner(
        PhilosopherDebate(["A", "B"], "issue", client, max_turns=20)
    )
    runner.start()
    time.sleep(0.15)
    runner.stop()

    assert _wait(runner)
    assert runner.stopped is True
    assert runner.result["stopped"] is True
    assert runner.result["turns"] < 20


def test_runner_captures_unexpected_worker_exceptions():
    class Exploding:
        def generate(self, prompt, system_prompt=""):
            raise RuntimeError("unexpected boom")

    runner = DebateRunner(PhilosopherDebate(["A"], "issue", Exploding(), max_turns=1))
    runner.start()

    assert _wait(runner)
    assert runner.result is None
    assert "unexpected boom" in runner.error


def test_runner_cannot_be_started_twice():
    runner = DebateRunner(PhilosopherDebate(["A"], "issue", SlowStreamingClient(["a"])))
    runner.start()
    assert _wait(runner)
    try:
        runner.start()
    except RuntimeError as e:
        assert "already been started" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError on second start")
