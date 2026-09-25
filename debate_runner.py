"""Run a PhilosopherDebate on a worker thread so the UI stays responsive.

The worker never touches Streamlit; it only mutates this object under a lock
and the Streamlit script polls :meth:`DebateRunner.snapshot`. That is what makes
a working Stop button possible, since Streamlit cannot deliver clicks while the
main script is blocked inside a long request.
"""

import threading

from debate_engine import DebateStopped


class DebateRunner:
    def __init__(self, debate):
        self.debate = debate
        self.result = None
        self.error = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._history: list[dict] = []
        self._streaming: tuple[str, str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("this runner has already been started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self.result = self.debate.run_debate(
                on_turn_callback=self._on_turn,
                on_chunk_callback=self._on_chunk,
                should_stop=self._stop.is_set,
            )
        except Exception as e:  # noqa: BLE001 - last-resort guard on a worker thread
            self.error = str(e)
        finally:
            self._finished.set()

    def _on_chunk(self, philosopher: str, text: str) -> None:
        if self._stop.is_set():
            raise DebateStopped
        with self._lock:
            self._streaming = (philosopher, text)

    def _on_turn(self, turn: int, philosopher: str, text: str) -> None:
        with self._lock:
            self._history = list(self.debate.history)
            self._streaming = None

    def stop(self) -> None:
        """Ask the worker to stop at the next chunk or turn boundary."""
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def is_running(self) -> bool:
        return not self._finished.is_set()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "history": list(self._history),
                "streaming": self._streaming,
                "result": self.result,
                "error": self.error,
            }
