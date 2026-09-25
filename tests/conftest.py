import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """Redirect the app's settings/debate files into a temporary directory."""
    monkeypatch.setenv(
        "MASTER_DEBATE_SETTINGS_FILE", str(tmp_path / "debate_settings.json")
    )
    monkeypatch.setenv("MASTER_DEBATE_DEBATES_DIR", str(tmp_path / "debates"))
    return tmp_path
