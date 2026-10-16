"""Independent component/coverage probes; never executes the agent graph."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LLM_LOGPROBS"] = "true"
from langchain_core.prompts import ChatPromptTemplate
from src.backend import LLMConfig, DEFAULT_OLLAMA_MODEL
from src import prompt_template as p
from src.invocation import invoke_component
from src.ollama_api import request
from src.telemetry import atomic_json, trajectory_context
from src.uncertainty import logprobs_coverage
from src.utils import PlanFormat, StepTaskFormat, QAAnswerFormat, PlanSummaryFormat


def run(output):
    if output.exists():
        raise FileExistsError(output)
    config = LLMConfig.from_env()
    if config.backend != "ollama" or config.model != DEFAULT_OLLAMA_MODEL or config.thinking:
        raise ValueError("This probe requires local qwen3:8b with thinking disabled")
    tag = next((t for t in request(config.base_url, "/api/tags")["models"] if t["name"] == config.model), None)
    if tag is None:
        raise FileNotFoundError("Exact model tag not installed; no model download")
    report = {"component_probes_only": True, "full_agent_graph_executed": False,
              "model": tag, "ollama": request(config.base_url, "/api/version"),
              "configuration": config.__dict__, "components": [], "coverage_controls": []}
    cases = [
        ("planner", p.planing_system_message, p.planing_human_message, {"question": "What is 1+1?", "memory": "empty"}, PlanFormat),
        ("step_definer", p.step_system_message, p.step_human_message, {"plan": "[Find the sum of 1+1]", "cur_step": "Find the sum of 1+1", "memory": ""}, StepTaskFormat),
        ("extractor", p.extract_system_messgage, p.extract_human_message, {"passage": "One plus one equals two.", "question": "What is 1+1?"}, None),
        ("qa", p.qa_system_message, p.qa_human_message, {"context": "doc_probe: [One plus one equals two.]", "question": "What is 1+1?"}, QAAnswerFormat),
        ("aggregate", p.aggregate_system_message, p.aggregate_human_message, {"question": "The sum of 1+1 was found to be 2. Return this sum."}, QAAnswerFormat),
        ("summary", p.summary_system_message, p.summary_human_message, {"question": "What is 1+1?", "plan": "[Find the sum]", "memory": "Task: Find the sum\nQuestion: What is 1+1?\nAnswer: 2\nConfident score: 8\n\n"}, PlanSummaryFormat),
    ]
    with trajectory_context("independent-component-probes", config, enabled=True,
                            output_root=output.parent/"component_probes", dataset_id="synthetic_component_inputs") as trace:
        trace.data["metadata"]["full_agent_graph_executed"] = False
        for component, system, human, inputs, schema in cases:
            entry = {"component": component}
            report["components"].append(entry)
            try:
                prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
                invoke_component(component, prompt, inputs, schema, step_id=f"probe:{component}",
                                 document_id="synthetic-doc" if component == "extractor" else None)
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            call = trace.data["invocations"][-1]
            entry.update({k: call.get(k) for k in ("status", "logprobs_status", "logprobs_count", "eval_count", "coverage_ratio", "coverage_status", "logprob_text_matches_content", "done_reason")})
            atomic_json(output, report)
            print(component, entry["status"], entry["logprobs_count"], entry["eval_count"], entry["coverage_status"], flush=True)
    report["trajectory_path"] = str(trace.path)
    base = {"model": config.model, "stream": False, "think": False, "logprobs": True}
    controls = [("plain_stop", "Reply exactly OK.", {"num_predict": 64}, None),
                ("plain_length_4", "Count from 1 to 100 separated by commas.", {"num_predict": 4}, None),
                ("plain_custom_stop", "Reply exactly A B C D E.", {"num_predict": 64, "stop": [" C"]}, None),
                ("structured_stop", 'Return answer="OK".', {"num_predict": 64}, {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]})]
    for name, prompt, options, schema in controls:
        payload = {**base, "messages": [{"role": "user", "content": prompt}],
                   "options": {"temperature": 0, "seed": 42, "num_ctx": config.num_ctx, **options}}
        if schema:
            payload["format"] = schema
        entry = {"name": name, "request": payload}
        try:
            raw = request(config.base_url, "/api/chat", payload)
            entry.update(raw_response=raw, **logprobs_coverage(raw, raw.get("message", {}).get("content", "")))
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        report["coverage_controls"].append(entry)
        atomic_json(output, report)
        print(name, entry.get("logprobs_count"), entry.get("eval_count"), entry.get("coverage_status"), flush=True)
    # API generate may expose context token IDs; preserve, do not re-tokenize
    # or treat a different request as a source of missing original logprobs.
    payload = {**base, "prompt": "Reply exactly OK.", "options": {"temperature": 0, "seed": 42, "num_ctx": config.num_ctx, "num_predict": 64}}
    raw = request(config.base_url, "/api/generate", payload)
    show = request(config.base_url, "/api/show", {"model": config.model})
    report["generate_control"] = {"request": payload, "raw_response": raw,
        "special_token_metadata": {k: v for k, v in show.get("model_info", {}).items() if k.endswith("token_id")},
        **logprobs_coverage(raw, raw.get("response", ""))}
    report["all_components_logprobs_available"] = all(e.get("logprobs_status") == "available" for e in report["components"])
    report["loaded_models"] = request(config.base_url, "/api/ps")
    atomic_json(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
