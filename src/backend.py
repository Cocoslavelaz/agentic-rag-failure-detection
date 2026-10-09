"""One configuration and factory for all six MA-RAG LLM components."""
from dataclasses import dataclass
import os
from urllib.parse import urlparse

TEMPERATURES = {"planner": 0.3, "step_definer": 0.3, "extractor": 0.0,
                "qa": 0.3, "aggregate": 0.0, "summary": 0.0}
DEFAULT_OLLAMA_MODEL = "qwen3:8b"


@dataclass(frozen=True)
class LLMConfig:
    backend: str = "ollama"
    model: str = DEFAULT_OLLAMA_MODEL
    base_url: str = "http://localhost:11434"
    thinking: bool = False
    num_ctx: int = 8192
    timeout: float = 180

    @classmethod
    def from_env(cls):
        backend = os.getenv("LLM_BACKEND", "ollama").lower()
        if backend not in {"ollama", "openai"}:
            raise ValueError(f"Unsupported LLM_BACKEND: {backend}")
        thinking = os.getenv("OLLAMA_THINKING", "false").lower()
        if thinking not in {"true", "false"}:
            raise ValueError("OLLAMA_THINKING must be true or false")
        model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL) if backend == "ollama" else os.getenv("MODEL_NAME")
        if not model:
            raise ValueError("MODEL_NAME is required for the optional OpenAI backend")
        config = cls(backend, model, os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                     thinking == "true", int(os.getenv("OLLAMA_NUM_CTX", "8192")),
                     float(os.getenv("LLM_TIMEOUT_SECONDS", "180")))
        if config.num_ctx <= 0 or config.timeout <= 0:
            raise ValueError("Context length and timeout must be positive")
        if backend == "ollama" and urlparse(config.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("This baseline permits only a local Ollama endpoint")
        return config

    @property
    def configuration_id(self):
        return "ma_rag_qwen3_8b" if self.backend == "ollama" and self.model == DEFAULT_OLLAMA_MODEL else f"ma_rag_{self.backend}_{self.model.replace(':', '_').replace('/', '_')}"


def build_llm(component, config=None):
    config = config or LLMConfig.from_env()
    temperature = TEMPERATURES[component]
    if config.backend == "ollama":
        from langchain_ollama import ChatOllama
        if "reasoning" not in ChatOllama.model_fields:
            raise RuntimeError("Installed ChatOllama cannot explicitly configure thinking; decision required")
        return ChatOllama(model=config.model, base_url=config.base_url,
                          temperature=temperature, reasoning=config.thinking,
                          num_ctx=config.num_ctx, client_kwargs={"timeout": config.timeout})
    from langchain_openai import ChatOpenAI
    key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not key:
        raise ValueError("OpenAI backend requires OPENAI_API_KEY (or legacy API_KEY)")
    kwargs = {} if component == "planner" else {"max_retries": 5}
    return ChatOpenAI(model=config.model, temperature=temperature, api_key=key,
                      timeout=config.timeout, **kwargs)
