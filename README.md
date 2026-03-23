# Master Debate

A Python application that orchestrates debates between multiple LLMs, each embodying a different philosopher's personality.

## Features

- **2-5 Philosophers** debate any given issue
- **20 Famous Philosophers** to choose from (Socrates, Plato, Aristotle, Kant, Nietzsche, etc.)
- **LLM Integration** with Ollama or compatible APIs
- **Consensus Detection** using a judgment-based approach
- **Time Limit** protection to prevent infinite debates
- **Streamlit Web UI** for easy interaction

## Setup

### 1. Install Dependencies

```bash
pip install streamlit requests
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
3. **Set Time Limit**: Configure debate duration (default 5 minutes)
4. **Start Debate**: WatchLLMs take turns presenting arguments
5. **Consensus Check**: After each turn, a judge model evaluates if consensus was reached
6. **Review Results**: See full debate history and summary

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
- **Model Names**: Available models in your Ollama instance
- **Max Turns**: 20 turns maximum per debate
- **Temperature**: 0.7 (balanced creativity/rationality)
- **Response Length**: 512 tokens per turn

## Tips

- **Model Quality**: Better philosophical reasoning with larger models (Llama3 70B, Phi-4, etc.)
- **Consensus**: Tough philosophical issues rarely reach consensus - that's the point!
- **Debate Quality**: Try contrasting philosophies (e.g., Rawls vs. Nozick on justice)

## License

MIT License - feel free to modify and extend!
