import pytest
from src.backend import DEFAULT_OLLAMA_MODEL, LLMConfig, TEMPERATURES, build_llm


def test_default_and_environment_configuration(monkeypatch):
    for name in ["OLLAMA_MODEL", "OLLAMA_BASE_URL", "OLLAMA_THINKING", "OLLAMA_NUM_CTX"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    config = LLMConfig.from_env()
    assert config.model == DEFAULT_OLLAMA_MODEL
    assert config.thinking is False and config.num_ctx == 8192
    assert config.configuration_id == "ma_rag_qwen3_8b"
    for component, temperature in TEMPERATURES.items():
        llm = build_llm(component, config)
        params = llm._chat_params([], stream=False, logprobs=True)
        assert params["think"] is False and params["logprobs"] is True
        assert params["options"].temperature == temperature
        assert params["options"].num_ctx == 8192
    monkeypatch.setenv("OLLAMA_MODEL", "custom:tag")
    assert LLMConfig.from_env().model == "custom:tag"


@pytest.mark.parametrize("key,value", [("LLM_BACKEND", "unknown"), ("OLLAMA_THINKING", "auto"),
    ("OLLAMA_NUM_CTX", "0"), ("LLM_TIMEOUT_SECONDS", "-1"), ("OLLAMA_BASE_URL", "https://remote.example")])
def test_bad_config_is_explicit(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        LLMConfig.from_env()


def test_optional_openai_factory_no_api_call(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "gpt-4o")
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(ValueError):
        build_llm("qa")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    llm = build_llm("qa")
    assert llm.model_name == "gpt-4o" and llm.max_retries == 5
