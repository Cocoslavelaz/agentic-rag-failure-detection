"""Explicit small compatibility probe, never a dataset/MA-RAG pipeline run."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
from src.backend import LLMConfig
from src.uncertainty import generated_logprobs, nll_from_logprobs
from src.ollama_api import request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = LLMConfig.from_env()
    tags = request(config.base_url, "/api/tags")
    tag = next((t for t in tags["models"] if t["name"] == config.model), None)
    if tag is None:
        raise RuntimeError("Exact configured model not installed; no pull or fallback")
    report = {"synthetic_probe_only": True, "configuration": config.__dict__,
              "version": request(config.base_url, "/api/version"), "tag": tag,
              "show": request(config.base_url, "/api/show", {"model": config.model}),
              "packages": {n: importlib.metadata.version(n) for n in
                           ("langchain-core", "langchain-ollama", "ollama", "pydantic")}, "probes": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(args.output)

    schema = {"type": "object", "properties": {"answer": {"type": "string"},
              "success": {"type": "string"}}, "required": ["answer", "success"]}
    for name, thinking, structured in [("plain_default_thinking", None, False),
                                      ("plain_thinking_disabled", False, False),
                                      ("structured_thinking_disabled", False, True)]:
        payload = {"model": config.model, "stream": False, "logprobs": True,
                   "messages": [{"role": "user", "content": 'Return the answer to 1+1. If JSON is requested, use answer="2" and success="Yes".'}],
                   "options": {"temperature": 0, "num_ctx": config.num_ctx, "num_predict": 256}}
        if thinking is not None:
            payload["think"] = thinking
        if structured:
            payload["format"] = schema
        entry = {"name": name, "request": payload}
        report["probes"].append(entry)
        save()
        start = time.perf_counter()
        try:
            raw = request(config.base_url, "/api/chat", payload)
            values, status = generated_logprobs(raw)
            entry.update(response=raw, logprobs_status=status,
                         nll=nll_from_logprobs(values, status=status),
                         thinking_present=bool(raw.get("message", {}).get("thinking")))
            if structured:
                entry["parsed"] = json.loads(raw["message"]["content"])
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["latency_seconds"] = time.perf_counter()-start
        save()
        print(name, entry.get("logprobs_status"), entry.get("error", "ok"), flush=True)
    from langchain_ollama import ChatOllama
    from pydantic import BaseModel
    class Probe(BaseModel):
        answer: str
        success: str
    entry = {"name": "chatollama_structured_thinking_disabled"}
    report["probes"].append(entry)
    try:
        llm = ChatOllama(model=config.model, base_url=config.base_url, temperature=0,
                         reasoning=False, num_ctx=config.num_ctx, num_predict=256,
                         client_kwargs={"timeout": 180})
        result = llm.with_structured_output(Probe, method="json_schema", include_raw=True).invoke(
            'Return answer="2" and success="Yes" for 1+1.', stream=False, logprobs=True)
        raw = result["raw"]
        values, status = generated_logprobs(raw.response_metadata)
        entry.update(raw=raw.model_dump(), parsed=result["parsed"].model_dump() if result["parsed"] else None,
                     parsing_error=str(result["parsing_error"]) if result["parsing_error"] else None,
                     logprobs_status=status, nll=nll_from_logprobs(values, status=status))
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
    # Verify the actual production call path. RunnableMap in include_raw=True
    # above drops invoke kwargs; direct ChatOllama native format does not.
    entry = {"name": "chatollama_native_format_thinking_disabled",
             "request_settings": {"think": False, "stream": False, "logprobs": True, "num_ctx": config.num_ctx}}
    report["probes"].append(entry)
    try:
        raw = llm.invoke('Return answer="2" and success="Yes" for 1+1.',
                         format=Probe.model_json_schema(), stream=False, logprobs=True)
        parsed = Probe.model_validate_json(raw.content)
        values, status = generated_logprobs(raw.response_metadata)
        entry.update(raw=raw.model_dump(), parsed=parsed.model_dump(), logprobs_status=status,
                     nll=nll_from_logprobs(values, status=status))
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
    report["loaded_models"] = request(config.base_url, "/api/ps")
    from langchain_core.prompts import ChatPromptTemplate
    from src import prompt_template as prompts
    from src.invocation import invoke_component
    from src.telemetry import trajectory_context
    from src.utils import QAAnswerFormat
    entry = {"name": "production_qa_schema_and_logging", "component_smoke_only": True}
    report["probes"].append(entry)
    try:
        prompt = ChatPromptTemplate.from_messages([("system", prompts.qa_system_message), ("human", prompts.qa_human_message)])
        with trajectory_context("synthetic-qa-compatibility", config, enabled=True,
                                output_root=args.output.parent / "component_smoke", dataset_id="synthetic_probe_only") as trace:
            result = invoke_component("qa", prompt, {"context": "doc_test: [1+1 equals 2.]", "question": "What is 1+1?"}, QAAnswerFormat)
        entry.update(parsed=result, trajectory_path=str(trace.path),
                     logprobs_status=trace.data["invocations"][0]["logprobs_status"],
                     nll=trace.data["invocations"][0]["nll"])
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"], capture_output=True, text=True)
    report["gpu_after_probe"] = gpu.stdout
    save()
    print(entry["name"], entry.get("logprobs_status"), entry.get("error", "ok"), flush=True)


if __name__ == "__main__":
    main()
