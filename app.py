import json
import os
import time
from datetime import UTC, datetime

import streamlit as st

from debate_engine import PhilosopherDebate
from llm_client import LLMClient
from philosophers import philosophers

# Single source of truth for the debate countdown default, in minutes.
DEFAULT_TIME_LIMIT_MINUTES = 1440


def format_time(seconds):
    """Format seconds as MM:SS"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def debate_outcome(result: dict):
    """Return (level, message) describing how a debate ended."""
    if result.get("consensus_reached"):
        return "success", "✅ **Consensus Reached!**"
    if result.get("time_limit_reached"):
        return "warning", "⏱️ **Time limit reached - No consensus**"
    if result.get("max_turns_reached"):
        return "warning", "⚠️ **No consensus reached within the turn limit**"
    if result.get("error"):
        return "error", f"❌ **Debate could not run:** {result['error']}"
    return "warning", "⚠️ **Debate ended without consensus**"


def render_debate_outcome(result: dict) -> None:
    """Render the debate outcome banner through Streamlit."""
    level, message = debate_outcome(result)
    getattr(st, level)(message)


# File locations are module-level so tests (and deployments) can redirect them,
# and so importing the app never writes to the filesystem.
SETTINGS_FILE = os.environ.get("MASTER_DEBATE_SETTINGS_FILE", "debate_settings.json")
DEBATES_DIR = os.environ.get("MASTER_DEBATE_DEBATES_DIR", "debates_data")


def load_settings():
    """Load settings from file if exists"""
    default_settings = {
        "base_url": "http://localhost:11434",
        "selected_model": "qwen3-coder-next:q8_0",
        "judge_model": "qwen3-coder-next:q8_0",
        "model_count": 3,
        "philosophers": [],
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


def display_debate_results(results):
    """Display debate results in the main panel"""
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

    # Show summary
    if results.get("result"):
        result = results["result"]
        render_debate_outcome(result)

        st.subheader("📊 Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Turns", result.get("turns", 0))
        with col2:
            st.metric("Philosophers", len(results.get("philosophers", [])))
        with col3:
            st.metric("Consensus", "Yes" if result.get("consensus_reached") else "No")

        st.subheader("📜 Full History")
        for entry in results.get("history", []):
            with st.expander(f"Turn {entry['turn'] + 1}: {entry['name']}"):
                st.markdown(entry["text"])

        # Conversation summary (we need to recreate the debate object to get the summary)
        # For simplicity, we'll skip the summary here since we don't have the debate object
        # In a real implementation, we might store the summary or compute it from history
        # But for now, we'll just show the history
        st.subheader("📝 Debate History")
        st.markdown("See the full history above for details.")


def main():
    st.set_page_config(page_title="Philosopher Debates", page_icon="⚖️", layout="wide")

    st.title("⚖️ Philosopher Debates")
    st.markdown(
        "Multiple LLMs with different philosophical personalities debate issues"
    )

    # Settings storage - loads from file if exists, otherwise defaults
    if "_settings" not in st.session_state:
        st.session_state["_settings"] = load_settings()

    # Initialize per-philosopher settings if not present
    if "_philosopher_settings" not in st.session_state:
        st.session_state["_philosopher_settings"] = {}

    settings = st.session_state["_settings"]
    philosopher_settings = st.session_state["_philosopher_settings"]

    # Always-visible timer display
    st.divider()
    time_limit_minutes = settings.get("time_limit", DEFAULT_TIME_LIMIT_MINUTES)

    # Create a placeholder for the timer that we can update
    timer_placeholder = st.empty()

    # Show initial timer value
    hours = time_limit_minutes // 60
    minutes = time_limit_minutes % 60
    if hours > 0:
        timer_display = f"{hours:02d}:{minutes:02d}:00"
    else:
        timer_display = f"00:{minutes:02d}:00"
    timer_placeholder.info(f"⏱️ Time limit: {timer_display}")

    # Load saved debates
    if "_debates" not in st.session_state:
        st.session_state["_debates"] = load_all_debates()

    # Split into main content and sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Configuration panel (will be collapsed by default)
        with st.expander("⚙️ Settings", expanded=False):
            # LLM Settings
            DEFAULT_PROVIDERS = [
                "http://localhost:11434",
                "http://localhost:8080",
                "http://localhost:3000",
                "http://localhost:11435",
            ]

            current_base_url = settings.get("base_url", "http://localhost:11434")

            # Determine if current URL is in default list or custom
            if current_base_url in DEFAULT_PROVIDERS:
                provider_url = current_base_url
            else:
                provider_url = "Custom..."

            provider_url = st.selectbox(
                "LLM Provider URL",
                options=DEFAULT_PROVIDERS + ["Custom..."],
                index=DEFAULT_PROVIDERS.index(provider_url)
                if provider_url in DEFAULT_PROVIDERS
                else len(DEFAULT_PROVIDERS),
                help="Select your LLM provider or choose Custom...",
                key="provider_select",
            )

            if provider_url == "Custom...":
                base_url = st.text_input(
                    "Custom API URL",
                    value=current_base_url,
                    help="Enter your custom LLM API URL",
                    key="base_url_input",
                )
            else:
                base_url = provider_url

            # Update settings if base_url changed
            if base_url != settings.get("base_url"):
                settings["base_url"] = base_url
                save_settings(settings)

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

            current_model = settings.get("selected_model", "qwen3-coder-next:q8_0")
            model_index = (
                DEFAULT_MODELS.index(current_model)
                if current_model in DEFAULT_MODELS
                else 0
            )

            selected_model = st.selectbox(
                "LLM Model",
                options=DEFAULT_MODELS,
                index=model_index,
                help="Select the model to use for all philosophers",
                key="model_select",
            )

            # Update settings if model changed
            if selected_model != settings.get("selected_model"):
                settings["selected_model"] = selected_model
                save_settings(settings)

            # Judge model for the consensus/summary calls. Defaults to the same
            # model as the philosophers but can be set differently so the verdict
            # does not come from one of the debaters' own clients.
            current_judge_model = settings.get(
                "judge_model", current_model
            )
            judge_index = (
                DEFAULT_MODELS.index(current_judge_model)
                if current_judge_model in DEFAULT_MODELS
                else 0
            )

            judge_model = st.selectbox(
                "Judge model (consensus/summary)",
                options=DEFAULT_MODELS,
                index=judge_index,
                help="Model that decides whether consensus was reached",
                key="judge_model_select",
            )

            if judge_model != settings.get("judge_model"):
                settings["judge_model"] = judge_model
                save_settings(settings)

            model_count = settings.get("model_count", 3)
            model_count = st.number_input(
                "Number of philosophers (2-5)",
                min_value=2,
                max_value=5,
                value=model_count,
                key="model_count_input",
            )

            # Update settings if model_count changed
            if model_count != settings.get("model_count"):
                settings["model_count"] = model_count
                save_settings(settings)

            st.divider()
            st.subheader("Select Philosophers")

            # Multi-select for philosophers
            phi_names = list(philosophers.keys())
            selected_phi_list = settings.get("philosophers", [])

            if selected_phi_list and len(selected_phi_list) > 0:
                default_phi = selected_phi_list[
                    : min(model_count, len(selected_phi_list))
                ]
            else:
                default_phi = phi_names[: min(model_count, len(phi_names))]

            selected_phi = st.multiselect(
                "Choose philosophers",
                options=phi_names,
                default=default_phi,
                key="philosopher_select",
            )

            # Reconcile the multiselect with the requested philosopher count.
            reconciled_phi = list(selected_phi)
            while len(reconciled_phi) < model_count:
                remaining = [p for p in phi_names if p not in reconciled_phi]
                if not remaining:
                    break
                reconciled_phi.append(remaining[0])
            reconciled_phi = reconciled_phi[:model_count]

            if reconciled_phi != selected_phi:
                # Keep the widget in sync with the actual debate roster instead
                # of silently debating with philosophers the user cannot see.
                st.session_state["philosopher_select"] = reconciled_phi
                st.rerun()

            selected_phi = reconciled_phi

            # Update settings if philosophers changed
            if selected_phi != settings.get("philosophers"):
                settings["philosophers"] = selected_phi
                save_settings(settings)

            # Individual philosopher settings (collapsed by default)
            if selected_phi:
                st.subheader("Philosopher LLM Settings")
                for i, philosopher in enumerate(selected_phi):
                    with st.expander(f"⚙️ {philosopher} Settings", expanded=False):
                        # Get or initialize settings for this philosopher
                        if philosopher not in philosopher_settings:
                            philosopher_settings[philosopher] = {
                                "base_url": settings.get(
                                    "base_url", "http://localhost:11434"
                                ),
                                "model": settings.get(
                                    "selected_model", "qwen3-coder-next:q8_0"
                                ),
                            }

                        phil_settings = philosopher_settings[philosopher]

                        # LLM Provider URL
                        current_base_url = phil_settings.get(
                            "base_url", "http://localhost:11434"
                        )
                        # Determine if current URL is in default list or custom
                        if current_base_url in DEFAULT_PROVIDERS:
                            phil_provider_url = current_base_url
                        else:
                            phil_provider_url = "Custom..."

                        phil_provider_url = st.selectbox(
                            "LLM Provider URL",
                            options=DEFAULT_PROVIDERS + ["Custom..."],
                            index=DEFAULT_PROVIDERS.index(phil_provider_url)
                            if phil_provider_url in DEFAULT_PROVIDERS
                            else len(DEFAULT_PROVIDERS),
                            help="Select your LLM provider or choose Custom...",
                            key=f"provider_select_{philosopher}_{i}",
                        )

                        if phil_provider_url == "Custom...":
                            phil_base_url = st.text_input(
                                "Custom API URL",
                                value=current_base_url,
                                help="Enter your custom LLM API URL",
                                key=f"base_url_input_{philosopher}_{i}",
                            )
                        else:
                            phil_base_url = phil_provider_url

                        # Update settings if base_url changed
                        if phil_base_url != phil_settings.get("base_url"):
                            phil_settings["base_url"] = phil_base_url

                        # LLM Model
                        current_model = phil_settings.get(
                            "model", "qwen3-coder-next:q8_0"
                        )
                        model_index = (
                            DEFAULT_MODELS.index(current_model)
                            if current_model in DEFAULT_MODELS
                            else 0
                        )

                        phil_selected_model = st.selectbox(
                            "LLM Model",
                            options=DEFAULT_MODELS,
                            index=model_index,
                            help="Select the model to use for this philosopher",
                            key=f"model_select_{philosopher}_{i}",
                        )

                        # Update settings if model changed
                        if phil_selected_model != phil_settings.get("model"):
                            phil_settings["model"] = phil_selected_model

            time_limit = settings.get("time_limit", DEFAULT_TIME_LIMIT_MINUTES)
            time_limit = st.number_input(
                "Time limit (minutes)",
                min_value=1,
                max_value=9999999,
                value=time_limit,
                key="time_limit_input",
            )

            # Update settings if time_limit changed
            if time_limit != settings.get("time_limit"):
                settings["time_limit"] = time_limit
                save_settings(settings)

        # Debate history panel
        st.divider()
        st.header("📜 Debate History")

        if not st.session_state["_debates"]:
            st.info("No saved debates yet")
        else:
            for i, debate in enumerate(st.session_state["_debates"]):
                # Create a short description for the debate
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

                philosophers_list = debate.get("philosophers", [])
                philosophers_str = ", ".join(philosophers_list)
                if len(philosophers_str) > 30:
                    philosophers_str = philosophers_str[:27] + "..."

                # Create a button for each debate
                if st.button(
                    f"💬 {issue}\n👥 {philosophers_str}\n🕒 {time_str}",
                    key=f"debate_{i}",
                    use_container_width=True,
                ):
                    # Load this debate
                    st.session_state.results = debate
                    st.rerun()

    # Main content area
    st.divider()

    # Check if we have a debate to display (from history)
    if st.session_state.get("results"):
        # Display the selected debate
        display_debate_results(st.session_state.results)

        # Add a button to clear the displayed debate
        if st.button("❌ Clear Displayed Debate"):
            st.session_state.results = None
            st.rerun()
    else:
        # Show the normal debate form
        # Issue input
        issue = st.text_area(
            "Enter the issue to debate:",
            placeholder="e.g., 'Is artificial intelligence capable of true consciousness?'",
            height=80,
        )

        # Start/Reset buttons
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button(
                "🚀 Start Debate", type="primary", use_container_width=True
            ) and not st.session_state.get("running", False):
                st.session_state.running = True
                st.rerun()

        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.clear()
                st.rerun()

        st.divider()

        # Initialize session state
        if "running" not in st.session_state:
            st.session_state.running = False
        if "results" not in st.session_state:
            st.session_state.results = None
        if "_debates" not in st.session_state:
            st.session_state["_debates"] = []

        # Run debate if triggered
        if st.session_state.running:
            if not issue.strip():
                # No st.rerun() here: rerunning would discard the warning before
                # the user ever sees it.
                st.warning("Please enter a debate issue")
                st.session_state.running = False

            elif len(selected_phi) < 2:
                st.warning("Please select at least 2 philosophers")
                st.session_state.running = False

            else:
                # Create LLM clients for each philosopher based on their individual settings
                llm_clients = {}
                for philosopher in selected_phi:
                    if philosopher in philosopher_settings:
                        phil_settings = philosopher_settings[philosopher]
                        llm_clients[philosopher] = LLMClient(
                            phil_settings.get(
                                "base_url",
                                settings.get("base_url", "http://localhost:11434"),
                            ),
                            phil_settings.get(
                                "model",
                                settings.get("selected_model", "qwen3-coder-next:q8_0"),
                            ),
                        )
                    else:
                        # Fallback to global settings
                        llm_clients[philosopher] = LLMClient(
                            settings.get("base_url", "http://localhost:11434"),
                            settings.get("selected_model", "qwen3-coder-next:q8_0"),
                        )

                # The judge runs the consensus/summary calls on a dedicated client
                # so the verdict is not tied to any single debater's model.
                judge_client = LLMClient(
                    settings.get("base_url", "http://localhost:11434"),
                    settings.get(
                        "judge_model",
                        settings.get("selected_model", "qwen3-coder-next:q8_0"),
                    ),
                )

                # Create a custom debate engine that uses per-philosopher LLM clients
                debate = PhilosopherDebate(
                    selected_phi,
                    issue,
                    llm_clients,
                    judge_client=judge_client,
                    time_limit_seconds=time_limit * 60,
                )

                # Create containers
                status_container = st.container()
                debate_container = st.container()
                summary_container = st.container()

                # Timer - always start fresh, otherwise a previous debate's
                # start time would make the new countdown begin at zero.
                start_time = time.time()
                st.session_state.start_time = start_time

                # Run debate
                with status_container:
                    progress_bar = st.progress(0)
                    timer_text = st.empty()

                def update_display(current_turn, philosopher, response):
                    # Update timer
                    elapsed = time.time() - start_time
                    remaining = max(0, time_limit * 60 - elapsed)
                    timer_text.info(
                        f"⏱️ Time remaining: {format_time(remaining)} | Turn: {current_turn + 1}/{debate.max_turns}"
                    )

                    # Update the always-visible timer display
                    hours = int(remaining // 3600)
                    minutes = int((remaining % 3600) // 60)
                    seconds = int(remaining % 60)
                    timer_display = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    timer_placeholder.info(f"⏱️ Time remaining: {timer_display}")

                    # Update progress
                    progress = min(1.0, (current_turn + 1) / debate.max_turns)
                    progress_bar.progress(progress)

                    # Add conversation entry
                    with debate_container:
                        st.markdown(f"### {philosopher}")
                        st.markdown(f"{response}")
                        st.divider()

                    st.session_state.last_message = response

                # Run the debate
                result = debate.run_debate(on_turn_callback=update_display)

                # Save results
                debate_data = {
                    "issue": issue,
                    "philosophers": selected_phi,
                    "result": result,
                    "history": debate.history,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "settings": {
                        "base_url": base_url,
                        "selected_model": selected_model,
                        "judge_model": settings.get("judge_model"),
                        "model_count": len(selected_phi),
                        "time_limit": time_limit,
                    },
                }

                st.session_state.results = debate_data

                # Save to file
                save_debate(debate_data)

                # Reload debates list
                st.session_state["_debates"] = load_all_debates()

                # Final timer update
                elapsed = time.time() - start_time
                timer_text.success(f"⏱️ Debate completed in {format_time(elapsed)}")

                # Reset timer to show the time limit
                time_limit_minutes = settings.get(
                    "time_limit", DEFAULT_TIME_LIMIT_MINUTES
                )
                hours = time_limit_minutes // 60
                minutes = time_limit_minutes % 60
                if hours > 0:
                    timer_display = f"{hours:02d}:{minutes:02d}:00"
                else:
                    timer_display = f"00:{minutes:02d}:00"
                timer_placeholder.info(f"⏱️ Time limit: {timer_display}")

                # Show summary
                with summary_container:
                    st.header("📊 Summary")

                    render_debate_outcome(result)

                    st.subheader("📊 Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Turns", result.get("turns", 0))
                    with col2:
                        st.metric("Philosophers", len(selected_phi))
                    with col3:
                        st.metric(
                            "Consensus",
                            "Yes" if result.get("consensus_reached") else "No",
                        )

                    st.subheader("📜 Full History")
                    for entry in debate.history:
                        with st.expander(f"Turn {entry['turn'] + 1}: {entry['name']}"):
                            st.markdown(entry["text"])

                    # Conversation summary
                    summary = debate.get_summary()
                    st.subheader("📝 Summary")
                    st.markdown(summary)

                st.session_state.running = False

            # Show previous debates if available (expanded by default for new debates)
            if st.session_state.results:
                with st.expander("📋 Previous Debate Results", expanded=False):
                    st.json(
                        {
                            "issue": st.session_state.results["issue"],
                            "philosophers": st.session_state.results["philosophers"],
                            "consensus": st.session_state.results["result"].get(
                                "consensus_reached", False
                            ),
                            "turns": st.session_state.results["result"].get("turns", 0),
                        }
                    )

    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Powered by Ollama/LLM API • Debate philosophical issues between AI philosophers"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
