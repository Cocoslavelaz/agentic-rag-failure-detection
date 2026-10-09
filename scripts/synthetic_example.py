"""Offline example through the real LangGraphs, with synthetic LLM/retrieval."""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
from langchain_core.messages import AIMessage
from agents.plan import plan_agent
from agents.plan_executor import build_plan_executor
from src.backend import LLMConfig
from src.telemetry import ACTIVE_TRACE, atomic_json, trajectory_context
from src.uncertainty import retrieval_features


class SyntheticLLM:
    def __init__(self, payload, requests, component, *, logprobs=True):
        self.payload, self.requests, self.component, self.logprobs = payload, requests, component, logprobs

    def invoke(self, messages, **kwargs):
        self.requests.append({"component": self.component, "messages": messages, "kwargs": kwargs})
        content = self.payload if self.component == "extractor" else json.dumps(self.payload)
        metadata = {"eval_count": 2, "prompt_eval_count": 12}
        if self.logprobs:
            metadata["logprobs"] = [{"token": "synthetic-token-1", "logprob": -0.2},
                                    {"token": "synthetic-token-2", "logprob": -0.4}]
        return AIMessage(content=content, response_metadata=metadata)


class SyntheticFactory:
    def __init__(self, *, qa_success="Yes", aggregate_success="True", logprobs=True):
        answer = lambda s: {"analysis": "Synthetic analysis", "answer": "2", "success": s, "rating": 8}
        self.queues = {"planner": [{"analysis": "synthetic", "step": ["Find a value", "Combine the result"]}],
                       "step_definer": [{"type": "question-answering", "task": "What is 1+1?"},
                                        {"type": "aggregate", "task": "The previous result is 2. Return 2."}],
                       "extractor": ["Document one says 2.", "Document two says 2."],
                       "qa": [answer(qa_success)], "aggregate": [answer(aggregate_success)],
                       "summary": [{"output": "Successful", "answer": "2", "score": 8}]}
        self.requests = []
        self.logprobs = logprobs

    def __call__(self, component, config=None):
        return SyntheticLLM(self.queues[component].pop(0), self.requests, component, logprobs=self.logprobs)


class SyntheticRetriever:
    top_k = 2

    def __init__(self):
        self.calls = []

    def __call__(self, query):
        self.calls.append(query)
        trace = ACTIVE_TRACE.get()
        if trace:
            scores = [0.9, 0.8]
            trace.retrieval({"synthetic": True, "query": query, "top_k_requested": 2, "top_k_returned": 2,
                             "ordered_document_ids": ["doc-a", "doc-b"], "raw_faiss_scores": scores,
                             "cosine_similarities": scores, "index_version": "synthetic_fixture",
                             "embedding_model": "synthetic_unit_vectors", **retrieval_features(scores, cosine_verified=True)})
        return ["One plus one is two.", "The sum of 1 and 1 is 2."], ["doc-a", "doc-b"]


def run_synthetic(output_root, *, enabled=True, qa_success="Yes", aggregate_success="True", logprobs=True):
    factory = SyntheticFactory(qa_success=qa_success, aggregate_success=aggregate_success, logprobs=logprobs)
    retriever = SyntheticRetriever()
    with patch("src.invocation.build_llm", factory):
        with trajectory_context("synthetic-question", LLMConfig(), enabled=enabled,
                                output_root=output_root, dataset_id="synthetic_only",
                                metadata={"synthetic": True, "model_calls": "mocked", "real_model_output": False}) as trace:
            planned = plan_agent({"original_question": "What is 1+1?", "past_exp": []})
            output = build_plan_executor(retriever).invoke({"original_question": "What is 1+1?", "plan": planned["plan"], "stop": False})
            if trace:
                trace.data["final_agent_output"] = output
    return output, trace, factory, retriever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    _, trace, _, _ = run_synthetic(ROOT / "outputs" / "synthetic")
    atomic_json(args.output, trace.data)
    print(args.output)
