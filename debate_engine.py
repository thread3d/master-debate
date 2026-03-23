import json
from typing import List, Dict
from philosophers import philosophers


class PhilosopherDebate:
    def __init__(self, philosophers: List[str], initial_issue: str, llm_client):
        self.philosophers = philosophers
        self.initial_issue = initial_issue
        # llm_client can be either a single LLMClient or a dict mapping philosopher names to LLMClients
        self.llm_client = llm_client
        self.history: List[Dict] = []
        self.max_turns = 20
        self.turn_count = 0

    def _get_llm_for_philosopher(self, philosopher_name: str):
        """Get the appropriate LLM client for a philosopher"""
        if isinstance(self.llm_client, dict):
            # Return the specific philosopher's LLM client, or fall back to a default if not found
            return self.llm_client.get(
                philosopher_name,
                self.llm_client.get(
                    list(self.llm_client.keys())[0] if self.llm_client else None
                ),
            )
        else:
            # Return the single LLM client for all philosophers
            return self.llm_client

    def generate_debate_turn(self, turn_number: int) -> str:
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
        debate_text = self._format_full_history()
        consensus_prompt = f"""Analyze the following debate and determine if the participants have reached a consensus or common understanding. 
 
Debate History:
{debate_text}

Has consensus been reached? Answer ONLY with 'YES' or 'NO'."""

        # Use the first philosopher's LLM client for consensus checking, or fall back to default
        llm_client = (
            self._get_llm_for_philosopher(self.philosophers[0])
            if self.philosophers
            else self.llm_client
        )

        response = llm_client.generate(consensus_prompt, "")
        if response:
            return response.strip().upper().startswith("YES")
        return False

    def _format_full_history(self) -> str:
        if not self.history:
            return f"Initial issue: {self.initial_issue}"
        return "\n".join(f"{e['name']}: {e['text']}" for e in self.history)

    def run_debate(self, on_turn_callback=None):
        self.turn_count = 0

        for turn in range(self.max_turns):
            response = self.generate_debate_turn(turn)

            if response:
                philosopher_name = self.philosophers[turn % len(self.philosophers)]
                self.history.append(
                    {"turn": turn, "name": philosopher_name, "text": response}
                )

                if on_turn_callback:
                    on_turn_callback(self.turn_count, philosopher_name, response)

                self.turn_count += 1

                if self.turn_count >= 2 and self.check_consensus():
                    return {"consensus_reached": True, "turns": self.turn_count}

        return {
            "consensus_reached": False,
            "turns": self.turn_count,
            "timelimit_reached": True,
        }

    def get_summary(self) -> str:
        if not self.history:
            return "No debate occurred."

        summary_prompt = f"Provide a concise summary of the key points and conclusions from this philosophical debate:\n\n{self._format_full_history()}\n\nSummary:"
        # Use the first philosopher's LLM client for summary generation, or fall back to default
        llm_client = (
            self._get_llm_for_philosopher(self.philosophers[0])
            if self.philosophers
            else self.llm_client
        )
        return llm_client.generate(summary_prompt, "") or "Debate completed."
