import json
import os
import time
from datetime import UTC, datetime

import streamlit as st

from debate_engine import PhilosopherDebate
from debate_runner import DebateRunner
from llm_client import LLMClient, list_models
from philosophers import philosophers

# Single source of truth for the debate countdown default, in minutes.
DEFAULT_TIME_LIMIT_MINUTES = 30
MIN_PHILOSOPHERS = 2
MAX_PHILOSOPHERS = 5
POLL_INTERVAL_SECONDS = 0.4

# File locations are module-level so tests (and deployments) can redirect them,
# and so importing the app never writes to the filesystem.
SETTINGS_FILE = os.environ.get("MASTER_DEBATE_SETTINGS_FILE", "debate_settings.json")
DEBATES_DIR = os.environ.get("MASTER_DEBATE_DEBATES_DIR", "debates_data")

CUSTOM_OPTION = "Custom…"

DEFAULT_PROVIDERS = [
    "http://localhost:11434",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://localhost:11435",
]

DEFAULT_MODELS = [
    "qwen3-coder-next:q8_0",
    "qwen2.5:14b",
    "llama3:70b",
    "phi4",
    "mistral",
    "gemma2",
    "deepseek-r1",
    "mixtral",
    "codellama",
    "yi",
    "dolphin-llama3",
    "nemotron-3-super:latest",
]


def format_time(seconds):
    """Format seconds as MM:SS"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def debate_outcome(result: dict):
    """Return (level, message) describing how a debate ended."""
    if result.get("consensus_reached"):
        return "success", "✅ **Consensus Reached!**"
    if result.get("stopped"):
        return "warning", f"⏹️ **Stopped after {result.get('turns', 0)} turn(s)**"
    if result.get("time_limit_reached"):
        return "warning", "⏱️ **Time limit reached — no consensus**"
    if result.get("max_turns_reached"):
        return "warning", "⚠️ **No consensus within the turn limit**"
    if result.get("error"):
        return "error", f"❌ **Debate stopped:** {result['error']}"
    return "warning", "⚠️ **Debate ended without consensus**"


def render_debate_outcome(result: dict) -> None:
    """Render the debate outcome banner through Streamlit."""
    level, message = debate_outcome(result)
    getattr(st, level)(message)


def load_settings():
    """Load settings from file if exists"""
    default_settings = {
        "base_url": "http://localhost:11434",
        "selected_model": "qwen3-coder-next:q8_0",
        "judge_model": "qwen3-coder-next:q8_0",
        "philosophers": [],
        "per_philosopher_overrides": False,
        "time_limit": DEFAULT_TIME_LIMIT_MINUTES,
    }

    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                loaded_settings = json.load(f)
                # Merge with defaults to ensure all keys exist
                for key, value in default_settings.items():
                    if key not in loaded_settings:
                        loaded_settings[key] = value
                return loaded_settings
        except (OSError, json.JSONDecodeError) as e:
            st.error(f"Error loading settings: {e}")
            return default_settings
    else:
        return default_settings


def save_settings(settings):
    """Save settings to file"""
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except (OSError, TypeError, ValueError) as e:
        st.error(f"Error saving settings: {e}")


# Debate storage functions
def save_debate(debate_data):
    """Save a debate to a JSON file in the debates directory"""
    timestamp = debate_data.get("timestamp", datetime.now(UTC).isoformat())
    # Create a safe filename from the timestamp
    safe_timestamp = timestamp.replace(":", "-").replace(".", "-")
    filename = f"debate_{safe_timestamp}.json"
    filepath = os.path.join(DEBATES_DIR, filename)
    try:
        os.makedirs(DEBATES_DIR, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(debate_data, f, indent=2)
    except (OSError, TypeError, ValueError) as e:
        st.error(f"Error saving debate: {e}")


def load_all_debates():
    """Load all debates from the debates directory, sorted by timestamp (newest first)"""
    debates = []
    if not os.path.exists(DEBATES_DIR):
        return debates

    for filename in os.listdir(DEBATES_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(DEBATES_DIR, filename)
            try:
                with open(filepath) as f:
                    debate_data = json.load(f)
                    debates.append(debate_data)
            except (OSError, json.JSONDecodeError) as e:
                st.error(f"Error loading debate {filename}: {e}")

    # Sort by timestamp (newest first)
    debates.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return debates


# --- configuration widgets --------------------------------------------------


def provider_selector(label, current_base_url, key):
    """Select an LLM provider, offering a Custom… escape hatch."""
    if current_base_url in DEFAULT_PROVIDERS:
        index = DEFAULT_PROVIDERS.index(current_base_url)
    else:
        index = len(DEFAULT_PROVIDERS)

    choice = st.selectbox(
        label,
        options=[*DEFAULT_PROVIDERS, CUSTOM_OPTION],
        index=index,
        key=key,
    )
    if choice == CUSTOM_OPTION:
        return st.text_input(
            f"{label} (custom URL)",
            value=current_base_url,
            key=f"{key}__custom",
        ).strip() or current_base_url
    return choice


def model_selector(label, options, current, key):
    """Select a model from the discovered options, or type a custom name."""
    choices = [*options, CUSTOM_OPTION]
    index = choices.index(current) if current in choices else choices.index(CUSTOM_OPTION)

    choice = st.selectbox(label, options=choices, index=index, key=key)
    if choice == CUSTOM_OPTION:
        typed = st.text_input(
            f"{label} (custom name)",
            value=current if current not in options else "",
            key=f"{key}__custom",
        ).strip()
        return typed or current
    return choice


def get_model_options(base_url):
    """Return (selectable models, models discovered from the server).

    The server list is fetched once per provider URL and cached, so a stopped
    backend does not add latency to every rerun.
    """
    if "_model_cache" not in st.session_state:
        st.session_state["_model_cache"] = {}
    cache = st.session_state["_model_cache"]

    if base_url not in cache:
        cache[base_url] = list_models(base_url)
    discovered = cache[base_url]

    options = list(DEFAULT_MODELS)
    for name in discovered:
        if name not in options:
            options.append(name)
    return options, discovered


# --- results rendering ------------------------------------------------------


def render_history(history):
    """Render completed turns (used while a debate is still streaming)."""
    for entry in history:
        st.markdown(f"### {entry['name']}")
        st.markdown(entry["text"])
        st.divider()


def render_debate_report(results):
    """Render the full report for a finished debate (live or from history)."""
    result = results.get("result") or {}

    render_debate_outcome(result)

    if result.get("failed_turns"):
        last_error = (result.get("errors") or [""])[-1]
        st.warning(
            f"⚠️ {result['failed_turns']} turn(s) failed to generate. "
            f"Last error: {last_error}"
        )
    elif result.get("judge_errors"):
        st.warning(f"⚠️ Consensus check failed: {result['judge_errors'][-1]}")

    st.subheader("📊 Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Turns", result.get("turns", 0))
    with col2:
        st.metric("Philosophers", len(results.get("philosophers", [])))
    with col3:
        st.metric("Consensus", "Yes" if result.get("consensus_reached") else "No")

    if results.get("elapsed_seconds"):
        st.caption(f"Finished in {format_time(results['elapsed_seconds'])}")

    st.subheader("📝 Summary")
    st.markdown(results.get("summary") or "_No summary was saved for this debate._")

    st.subheader("📜 Full History")
    for entry in results.get("history", []):
        with st.expander(f"Turn {entry['turn'] + 1}: {entry['name']}"):
            st.markdown(entry["text"])


def display_debate_results(results):
    """Display a saved debate in the main panel."""
    if not results:
        return

    st.divider()
    st.subheader(f"Issue: {results['issue']}")

    # Current records nest the LLM settings under "settings"; older ones stored
    # them at the top level. Support both shapes so history playback never
    # raises a KeyError.
    debate_settings = results.get("settings") or results
    st.caption(
        f"Settings: {debate_settings.get('base_url', 'unknown')} | "
        f"{debate_settings.get('selected_model', 'unknown')} | "
        f"{debate_settings.get('model_count', len(results.get('philosophers', [])))} philosophers | "
        f"{debate_settings.get('time_limit', 'unknown')} min"
    )

    render_debate_report(results)


# --- running a debate -------------------------------------------------------


def build_clients(selected_phi, philosopher_settings, settings):
    """Create the per-philosopher clients plus a dedicated judge client."""
    base_url = settings.get("base_url", "http://localhost:11434")
    default_model = settings.get("selected_model", "qwen3-coder-next:q8_0")
    use_overrides = bool(settings.get("per_philosopher_overrides"))

    llm_clients = {}
    for philosopher in selected_phi:
        override = philosopher_settings.get(philosopher) if use_overrides else None
        if override:
            llm_clients[philosopher] = LLMClient(
                override.get("base_url", base_url),
                override.get("model", default_model),
            )
        else:
            llm_clients[philosopher] = LLMClient(base_url, default_model)

    judge_client = LLMClient(base_url, settings.get("judge_model", default_model))
    return llm_clients, judge_client


def run_live_debate(issue, selected_phi, settings, philosopher_settings, time_limit, timer_placeholder):
    """Start (or continue polling) the current debate, then render its report."""
    runner = st.session_state.get("_runner")

    if runner is None:
        llm_clients, judge_client = build_clients(
            selected_phi, philosopher_settings, settings
        )
        debate = PhilosopherDebate(
            selected_phi,
            issue,
            llm_clients,
            judge_client=judge_client,
            time_limit_seconds=time_limit * 60,
        )
        runner = DebateRunner(debate)
        st.session_state["_runner"] = runner
        st.session_state["_debate_start"] = time.time()
        runner.start()

    start_time = st.session_state.get("_debate_start", time.time())
    status_container = st.container()
    debate_container = st.container()

    with status_container:
        progress_bar = st.progress(0, text="Starting debate…")
        elapsed = time.time() - start_time
        remaining = max(0, time_limit * 60 - elapsed)
        timer_placeholder.info(f"⏱️ Time remaining: {format_time(remaining)}")

    snapshot = runner.snapshot()
    turn_count = len(snapshot["history"])
    max_turns = runner.debate.max_turns

    with debate_container:
        render_history(snapshot["history"])
        if snapshot["streaming"]:
            philosopher, partial = snapshot["streaming"]
            st.markdown(f"### {philosopher} _…_")
            st.markdown(partial)

    if runner.is_running():
        progress = min(1.0, turn_count / max(1, max_turns))
        progress_bar.progress(
            progress, text=f"Turn {turn_count}/{max_turns} — debate in progress"
        )
        if st.button("⏹ Stop debate", key="stop_debate"):
            runner.stop()
        time.sleep(POLL_INTERVAL_SECONDS)
        st.rerun()
        return

    # The worker has finished: persist and rerun into the report view.
    st.session_state["_runner"] = None
    st.session_state.running = False

    result = runner.result
    if result is None:
        result = {
            "consensus_reached": False,
            "error": runner.error or "the debate worker stopped unexpectedly",
        }
    if runner.stopped:
        result.setdefault("stopped", True)

    debate_data = {
        "issue": issue,
        "philosophers": list(selected_phi),
        "result": result,
        "history": list(runner.debate.history),
        "summary": runner.debate.get_summary(),
        "elapsed_seconds": time.time() - start_time,
        "timestamp": datetime.now(UTC).isoformat(),
        "settings": {
            "base_url": settings.get("base_url", "http://localhost:11434"),
            "selected_model": settings.get("selected_model"),
            "judge_model": settings.get("judge_model"),
            "model_count": len(selected_phi),
            "time_limit": time_limit,
        },
    }

    st.session_state.results = debate_data
    save_debate(debate_data)
    st.session_state["_debates"] = load_all_debates()
    st.rerun()


def render_sidebar(settings, philosopher_settings):
    """Render the configuration plus history sidebar; returns the live values."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        with st.expander("⚙️ Settings", expanded=False):
            current_base_url = settings.get("base_url", "http://localhost:11434")
            base_url = provider_selector(
                "LLM Provider URL", current_base_url, "provider_select"
            )
            if base_url != settings.get("base_url"):
                settings["base_url"] = base_url
                save_settings(settings)

            model_options, discovered = get_model_options(base_url)
            if discovered:
                st.caption(f"{len(discovered)} model(s) available on this server")
            else:
                st.caption("Server model list unavailable — using built-in names")
            if st.button("🔄 Refresh model list", key="refresh_models"):
                st.session_state.get("_model_cache", {}).pop(base_url, None)
                st.rerun()

            selected_model = model_selector(
                "LLM Model",
                model_options,
                settings.get("selected_model", DEFAULT_MODELS[0]),
                "model_select",
            )
            if selected_model != settings.get("selected_model"):
                settings["selected_model"] = selected_model
                save_settings(settings)

            judge_model = model_selector(
                "Judge model (consensus/summary)",
                model_options,
                settings.get("judge_model", selected_model),
                "judge_model_select",
            )
            if judge_model != settings.get("judge_model"):
                settings["judge_model"] = judge_model
                save_settings(settings)

            overrides = st.checkbox(
                "Override provider/model per philosopher",
                value=bool(settings.get("per_philosopher_overrides")),
                key="per_philosopher_overrides",
                help="Off by default — every philosopher then uses the settings above.",
            )
            if overrides != settings.get("per_philosopher_overrides"):
                settings["per_philosopher_overrides"] = overrides
                save_settings(settings)

            st.divider()
            st.subheader("Select Philosophers")

            phi_names = list(philosophers.keys())
            saved_phi = [p for p in settings.get("philosophers", []) if p in phi_names]
            default_phi = saved_phi[:MAX_PHILOSOPHERS] or phi_names[:3]

            selected_phi = st.multiselect(
                f"Choose philosophers ({MIN_PHILOSOPHERS}-{MAX_PHILOSOPHERS})",
                options=phi_names,
                default=default_phi,
                max_selections=MAX_PHILOSOPHERS,
                key="philosopher_select",
            )
            if selected_phi != settings.get("philosophers"):
                settings["philosophers"] = selected_phi
                save_settings(settings)

            if overrides and selected_phi:
                st.caption("Per-philosopher overrides")
                for i, philosopher in enumerate(selected_phi):
                    with st.expander(f"⚙️ {philosopher}", expanded=False):
                        if philosopher not in philosopher_settings:
                            philosopher_settings[philosopher] = {
                                "base_url": base_url,
                                "model": selected_model,
                            }
                        phil = philosopher_settings[philosopher]

                        phil_url = provider_selector(
                            "Provider", phil.get("base_url", base_url), f"phil_provider_{i}"
                        )
                        if phil_url != phil.get("base_url"):
                            phil["base_url"] = phil_url

                        phil_options, _ = get_model_options(phil_url)
                        phil_model = model_selector(
                            "Model",
                            phil_options,
                            phil.get("model", selected_model),
                            f"phil_model_{i}",
                        )
                        if phil_model != phil.get("model"):
                            phil["model"] = phil_model

            time_limit = st.number_input(
                "Time limit (minutes)",
                min_value=1,
                max_value=1440,
                value=settings.get("time_limit", DEFAULT_TIME_LIMIT_MINUTES),
                key="time_limit_input",
            )
            if time_limit != settings.get("time_limit"):
                settings["time_limit"] = time_limit
                save_settings(settings)

        st.divider()
        st.header("📜 Debate History")

        if not st.session_state["_debates"]:
            st.info("No saved debates yet")
        else:
            for i, debate in enumerate(st.session_state["_debates"]):
                issue = debate.get("issue", "Unknown Issue")[:50]
                if len(debate.get("issue", "")) > 50:
                    issue += "..."

                timestamp = debate.get("timestamp", "")
                try:
                    # Format timestamp in the viewer's local time.
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    time_str = dt.astimezone().strftime("%m/%d %H:%M")
                except (TypeError, ValueError):
                    time_str = timestamp[:16] if timestamp else "Unknown time"

                philosophers_str = ", ".join(debate.get("philosophers", []))
                if len(philosophers_str) > 30:
                    philosophers_str = philosophers_str[:27] + "..."

                if st.button(
                    f"💬 {issue}\n👥 {philosophers_str}\n🕒 {time_str}",
                    key=f"debate_{i}",
                    use_container_width=True,
                ):
                    st.session_state.results = debate
                    st.rerun()

    return selected_phi, time_limit


def main():
    st.set_page_config(page_title="Philosopher Debates", page_icon="⚖️", layout="wide")

    st.title("⚖️ Philosopher Debates")
    st.markdown(
        "Multiple LLMs with different philosophical personalities debate issues"
    )

    if "_settings" not in st.session_state:
        st.session_state["_settings"] = load_settings()
    if "_philosopher_settings" not in st.session_state:
        st.session_state["_philosopher_settings"] = {}
    if "_debates" not in st.session_state:
        st.session_state["_debates"] = load_all_debates()
    if "running" not in st.session_state:
        st.session_state.running = False
    if "results" not in st.session_state:
        st.session_state.results = None

    settings = st.session_state["_settings"]
    philosopher_settings = st.session_state["_philosopher_settings"]

    st.divider()
    time_limit_minutes = settings.get("time_limit", DEFAULT_TIME_LIMIT_MINUTES)
    timer_placeholder = st.empty()
    hours = time_limit_minutes // 60
    minutes = time_limit_minutes % 60
    timer_display = (
        f"{hours:02d}:{minutes:02d}:00" if hours > 0 else f"00:{minutes:02d}:00"
    )
    timer_placeholder.info(f"⏱️ Time limit: {timer_display}")

    selected_phi, time_limit = render_sidebar(settings, philosopher_settings)

    st.divider()

    if st.session_state.get("results"):
        display_debate_results(st.session_state.results)

        if st.button("❌ Clear Displayed Debate"):
            st.session_state.results = None
            st.rerun()
    else:
        issue = st.text_area(
            "Enter the issue to debate:",
            placeholder="e.g., 'Is artificial intelligence capable of true consciousness?'",
            height=80,
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(
                "🚀 Start Debate", type="primary", use_container_width=True
            ) and not st.session_state.get("running", False):
                st.session_state.running = True
                st.rerun()
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                runner = st.session_state.get("_runner")
                if runner is not None:
                    runner.stop()
                st.session_state.clear()
                st.rerun()

        st.divider()

        if st.session_state.running:
            if not issue.strip():
                st.warning("Please enter a debate issue")
                st.session_state.running = False
            elif len(selected_phi) < MIN_PHILOSOPHERS:
                st.warning(f"Please select at least {MIN_PHILOSOPHERS} philosophers")
                st.session_state.running = False
            else:
                run_live_debate(
                    issue,
                    selected_phi,
                    settings,
                    philosopher_settings,
                    time_limit,
                    timer_placeholder,
                )

    st.divider()
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Powered by Ollama/LLM API • Debate philosophical issues between AI philosophers"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
