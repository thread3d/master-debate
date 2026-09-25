import time

from llm_client import LLMError
from philosophers import philosophers


class DebateStopped(Exception):
    """Raised by a chunk callback to abort a debate mid-turn."""


class PhilosopherDebate:
    def __init__(
        self,
        philosophers: list[str],
        initial_issue: str,
        llm_client,
        judge_client=None,
        max_turns: int = 20,
        time_limit_seconds: float | None = None,
        max_consecutive_failures: int = 2,
    ):
        self.philosophers = list(philosophers)
        self.initial_issue = initial_issue
        # llm_client can be either a single LLMClient or a dict mapping philosopher names to LLMClients
        self.llm_client = llm_client
        # judge_client handles the meta-level calls (consensus, summary). When it is
        # not supplied we fall back to one of the debating philosophers' clients.
        self.judge_client = judge_client
        self.history: list[dict] = []
        self.max_turns = max_turns
        self.time_limit_seconds = time_limit_seconds
        # A backend that is down should fail fast rather than silently burning
        # through every turn, so we abort after this many failures in a row.
        self.max_consecutive_failures = max_consecutive_failures
        self.turn_count = 0
        self.errors: list[str] = []
        self.judge_errors: list[str] = []

    def _get_llm_for_philosopher(self, philosopher_name: str):
        """Get the appropriate LLM client for a philosopher.

        Accepts either a single client shared by all philosophers or a mapping
        of philosopher name -> client. When a mapping has no entry for the
        requested name, the first available client is used as a fallback.
        Returns None only when no client is configured at all.
        """
        if isinstance(self.llm_client, dict):
            if philosopher_name in self.llm_client:
                return self.llm_client[philosopher_name]
            if self.llm_client:
                return next(iter(self.llm_client.values()))
            return None
        else:
            # Return the single LLM client for all philosophers
            return self.llm_client

    def _get_judge_client(self):
        """Client used for consensus/summary calls.

        An explicitly injected judge client always wins; otherwise fall back to
        the first philosopher's client (or the shared client).
        """
        if self.judge_client is not None:
            return self.judge_client
        if self.philosophers:
            return self._get_llm_for_philosopher(self.philosophers[0])
        return self._get_llm_for_philosopher("")

    def _time_limit_exceeded(self, start_time: float) -> bool:
        if self.time_limit_seconds is None:
            return False
        return (time.monotonic() - start_time) >= self.time_limit_seconds

    def _failure_info(self) -> dict:
        """Extra result fields describing any generation failures."""
        info = {}
        if self.errors:
            info["failed_turns"] = len(self.errors)
            info["errors"] = list(self.errors)
        if self.judge_errors:
            info["judge_errors"] = list(self.judge_errors)
        return info

    def generate_debate_turn(self, turn_number: int, on_chunk=None) -> str:
        """Generate one turn.

        When an ``on_chunk`` callback is supplied and the client supports
        streaming, partial text is delivered as it arrives. Raises LLMError if
        the backend call fails and DebateStopped if the callback aborts.
        """
        if not self.philosophers:
            return ""

        philosopher_name = self.philosophers[turn_number % len(self.philosophers)]
        philosopher_data = philosophers.get(philosopher_name, {})
        system_prompt = philosopher_data.get("system", "")
        personality = philosopher_data.get("temperament", "")

        context = self._build_context(turn_number)

        prompt = f"""You are {philosopher_name}, engaging in a philosophical debate about: "{self.initial_issue}"

        Your personality: {personality}

        Previous debate context:
        {context}

        Your turn - respond to the debate while staying true to your philosophical perspective. Be concise but substantive.

        {philosopher_name}:"""

        llm_client = self._get_llm_for_philosopher(philosopher_name)
        if llm_client is None:
            return ""

        if on_chunk is not None and hasattr(llm_client, "stream_generate"):
            text = ""
            for partial in llm_client.stream_generate(prompt, system_prompt):
                text = partial
                on_chunk(philosopher_name, partial)
            return text

        return llm_client.generate(prompt, system_prompt) or ""

    def _build_context(self, current_turn: int) -> str:
        if current_turn == 0:
            return f"Initial issue: {self.initial_issue}\n\nThe debate begins now."

        excerpts = []
        for entry in self.history[-6:]:
            name = entry.get("name", "Unknown")
            text = entry.get("text", "")
            excerpts.append(f"{name}: {text}")

        return "\n\n".join(excerpts)

    def check_consensus(self) -> bool:
        if not self.history or not self.philosophers:
            return False

        debate_text = self._format_full_history()
        consensus_prompt = f"""Analyze the following debate and determine if the participants have reached a consensus or common understanding. 
 
 Debate History:
 {debate_text}

 Has consensus been reached? Answer ONLY with 'YES' or 'NO'."""

        # The judge is used instead of a debating philosopher so the verdict is
        # not biased by whichever client happens to be first.
        llm_client = self._get_judge_client()
        if llm_client is None:
            return False

        try:
            response = llm_client.generate(consensus_prompt, "")
        except LLMError as e:
            # A judge failure should not end the debate; the participants can
            # keep talking and we simply cannot confirm consensus.
            self.judge_errors.append(str(e))
            return False

        if response:
            return response.strip().upper().startswith("YES")
        return False

    def _format_full_history(self) -> str:
        if not self.history:
            return f"Initial issue: {self.initial_issue}"
        return "\n".join(f"{e['name']}: {e['text']}" for e in self.history)

    def run_debate(self, on_turn_callback=None, on_chunk_callback=None, should_stop=None):
        self.turn_count = 0
        self.errors = []
        self.judge_errors = []
        start_time = time.monotonic()

        if not self.philosophers:
            return {
                "consensus_reached": False,
                "turns": 0,
                "error": "no philosophers selected",
            }

        consecutive_failures = 0

        for turn in range(self.max_turns):
            if should_stop is not None and should_stop():
                return {
                    "consensus_reached": False,
                    "turns": self.turn_count,
                    "stopped": True,
                    **self._failure_info(),
                }

            if self._time_limit_exceeded(start_time):
                return {
                    "consensus_reached": False,
                    "turns": self.turn_count,
                    "time_limit_reached": True,
                    **self._failure_info(),
                }

            try:
                response = self.generate_debate_turn(turn, on_chunk=on_chunk_callback)
            except DebateStopped:
                return {
                    "consensus_reached": False,
                    "turns": self.turn_count,
                    "stopped": True,
                    **self._failure_info(),
                }
            except LLMError as e:
                self.errors.append(str(e))
                consecutive_failures += 1
                if consecutive_failures >= self.max_consecutive_failures:
                    return {
                        "consensus_reached": False,
                        "turns": self.turn_count,
                        "error": str(e),
                        **self._failure_info(),
                    }
                continue
            else:
                consecutive_failures = 0

            if response:
                philosopher_name = self.philosophers[turn % len(self.philosophers)]
                self.history.append(
                    {"turn": turn, "name": philosopher_name, "text": response}
                )

                if on_turn_callback:
                    on_turn_callback(self.turn_count, philosopher_name, response)

                self.turn_count += 1

                if self.turn_count >= 2 and self.check_consensus():
                    return {
                        "consensus_reached": True,
                        "turns": self.turn_count,
                        **self._failure_info(),
                    }

                if should_stop is not None and should_stop():
                    return {
                        "consensus_reached": False,
                        "turns": self.turn_count,
                        "stopped": True,
                        **self._failure_info(),
                    }

                # Consensus checks consume wall-clock time too, so re-check the
                # deadline before starting another turn.
                if self._time_limit_exceeded(start_time):
                    return {
                        "consensus_reached": False,
                        "turns": self.turn_count,
                        "time_limit_reached": True,
                        **self._failure_info(),
                    }

        return {
            "consensus_reached": False,
            "turns": self.turn_count,
            "max_turns_reached": True,
            **self._failure_info(),
        }

    def get_summary(self) -> str:
        if not self.history:
            return "No debate occurred."

        summary_prompt = f"Provide a concise summary of the key points and conclusions from this philosophical debate:\n\n{self._format_full_history()}\n\nSummary:"
        llm_client = self._get_judge_client()
        if llm_client is None:
            return "Debate completed."
        try:
            return llm_client.generate(summary_prompt, "") or "Debate completed."
        except LLMError as e:
            self.judge_errors.append(str(e))
            return f"Summary unavailable: {e}"
