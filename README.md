# Master Debate

A Python application that orchestrates debates between multiple LLMs, each embodying a different philosopher's personality.

## Features

- **2-5 Philosophers** debate any given issue
- **20 Famous Philosophers** to choose from (Socrates, Plato, Aristotle, Kant, Nietzsche, etc.)
- **LLM Integration** with Ollama or compatible APIs
- **Automatic Model Discovery** from `/api/tags`, plus a Custom option for any name
- **Streaming Output** — responses appear as they are generated, with a working Stop button
- **Consensus Detection** using a judgment-based approach
- **Independent Judge Model** for consensus checks and summaries
- **Time & Turn Limits** — the debate stops at the configured duration or after 20 turns
- **Clear Failure Reporting** — an unreachable backend or a missing model is shown, not swallowed
- **Streamlit Web UI** for easy interaction

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup LLM Backend

Ensure you have an LLM API server running. Options:

#### Ollama (Recommended)
```bash
# Install Ollama from https://ollama.com
ollama pull llama3
ollama pull phi4
```

#### Compatible API Endpoints
The app expects endpoints at `/api/generate` and `/api/chat` (Ollama-compatible format).

### 3. Run the App

```bash
streamlit run app.py
```

## How It Works

1. **Select Philosophers**: Choose 2-5 philosophers from the dropdown
2. **Enter Issue**: Type the philosophical issue to debate
3. **Set Time Limit**: Configure how long the debate may run (default 30 minutes)
4. **Choose Models**: Pick from the models discovered on your server, or type a custom name
5. **Choose a Judge**: Optionally pick a separate model for the consensus/summary calls
6. **Start Debate**: Watch the arguments stream in turn by turn; press Stop to end early
7. **Consensus Check**: After each turn, the judge model evaluates if consensus was reached
8. **Review Results**: See the outcome, statistics, summary and full history

## Philosophy Personalities

Each philosopher has a unique personality defined by their core philosophical principles:

- **Socrates**: Dialectical questioning, "knowing that I know nothing"
- **Plato**: Theory of Forms, analogies and allegories
- **Aristotle**: Empirical observation, logical causality
- **Descartes**: Radical doubt, "I think therefore I am"
- **Nietzsche**: Will to Power, critique of morality
- **Rawls**: Veil of ignorance, fairness as justice
- **And 10+ more!**

## Configuration

- **LLM API URL**: Default `http://localhost:11434`
- **Model Names**: Discovered from the provider's `/api/tags`; if the server is
  unreachable the built-in list is used, and *Custom…* accepts any name
- **Judge Model**: Model used for the consensus and summary calls (defaults to the selected LLM model)
- **Per-philosopher overrides**: Off by default; enables a provider/model per philosopher
- **Time Limit**: The debate stops once this duration elapses (default 30 minutes)
- **Max Turns**: 20 turns maximum per debate
- **Temperature**: 0.7 (balanced creativity/rationality)
- **Response Length**: 512 tokens per turn
- **Python**: 3.11 or newer

## Tips

- **Model Quality**: Better philosophical reasoning with larger models (Llama3 70B, Phi-4, etc.)
- **Consensus**: Tough philosophical issues rarely reach consensus - that's the point!
- **Debate Quality**: Try contrasting philosophies (e.g., Rawls vs. Nozick on justice)

## Testing

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

The suite covers the debate engine, the streaming client, the background debate
runner, the philosopher roster, and the Streamlit app (via
`streamlit.testing.v1.AppTest`), so it runs without a network connection or an
LLM backend.

## License

MIT License - feel free to modify and extend!
