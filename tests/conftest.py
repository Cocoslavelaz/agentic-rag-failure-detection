import os
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["LLM_BACKEND"] = "ollama"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Offline unit tests must not make model/network calls")
    monkeypatch.setattr("urllib.request.urlopen", blocked)
    monkeypatch.setattr("ollama.Client.chat", blocked)
