import json
from pathlib import Path
import pytest
from langchain_core.prompts import ChatPromptTemplate
from scripts.synthetic_example import SyntheticFactory, SyntheticLLM, SyntheticRetriever, run_synthetic
from src.backend import LLMConfig
from src.invocation import invoke_component
from src.success import SuccessValidationError
from src.telemetry import trajectory_context
from src.utils import QAAnswerFormat


def test_multidocument_extractor_and_branch_alignment(tmp_path):
    output, trace, factory, retriever = run_synthetic(tmp_path)
    calls = trace.data["invocations"]
    assert [c["component"] for c in calls] == ["planner", "step_definer", "extractor", "extractor", "qa", "step_definer", "aggregate", "summary"]
    extracted = [c for c in calls if c["component"] == "extractor"]
    assert [c["document_id"] for c in extracted] == ["doc-a", "doc-b"]
    assert [c["retrieval_rank"] for c in extracted] == [1, 2]
    assert all(c["step_id"] == "step:0" for c in extracted)
    assert len(retriever.calls) == 1 and len(trace.data["retrievals"]) == 1
    steps = trace.data["steps"]
    assert steps[0]["execution_masks"] == {"retriever": True, "extractor": True, "qa": True, "aggregate": False}
    assert steps[1]["execution_masks"] == {"retriever": False, "extractor": False, "qa": False, "aggregate": True}
    assert output["step_output"][1]["success_raw"] == "True"
    assert output["step_output"][1]["success_normalized"] == "yes"
    assert len({c["invocation_id"] for c in calls}) == len(calls)
    assert all(c["status"] == "completed" for c in calls)
    for req in factory.requests:
        assert req["kwargs"]["logprobs"] is True and req["kwargs"]["stream"] is False
        if req["component"] != "extractor":
            assert "format" in req["kwargs"]
    assert json.loads(trace.path.read_text(encoding="utf-8")) == trace.data
    assert not trace.path.with_suffix(".json.tmp").exists()


def test_logging_disabled_same_outputs_and_calls(tmp_path):
    enabled, trace, first, _ = run_synthetic(tmp_path / "on")
    disabled, no_trace, second, _ = run_synthetic(tmp_path / "off", enabled=False)
    assert enabled == disabled and first.requests == second.requests
    assert no_trace is None and not (tmp_path / "off").exists()


@pytest.mark.parametrize("branch", ["qa", "aggregate"])
@pytest.mark.parametrize("raw,normalized", [(v, "yes") for v in ["Yes", "YES", "yes", "True", "true", True, "1", 1, " YES ", "\tTrue\n"]]
    + [(v, "no") for v in ["No", "NO", "no", "False", "false", False, "0", 0, " NO ", "\tfalse\n"]])
def test_actual_branches_normalize_and_route(tmp_path, branch, raw, normalized):
    kwargs = {f"{branch}_success": raw} if branch == "qa" else {"aggregate_success": raw}
    output, trace, _, _ = run_synthetic(tmp_path, **kwargs)
    answers = output["step_output"]
    answer = answers[0] if branch == "qa" else answers[-1]
    assert answer["success_raw"] == raw and type(answer["success_raw"]) is type(raw)
    assert answer["success_normalized"] == normalized
    assert len(answers) == (1 if branch == "qa" and normalized == "no" else 2)
    assert output["stop"] is True


@pytest.mark.parametrize("branch", ["qa", "aggregate"])
def test_invalid_success_keeps_raw_error_and_no_retry(tmp_path, branch):
    kwargs = {"qa_success": "maybe"} if branch == "qa" else {"aggregate_success": "maybe"}
    with pytest.raises(SuccessValidationError):
        run_synthetic(tmp_path, **kwargs)
    data = json.loads(next(tmp_path.rglob("trajectory.json")).read_text(encoding="utf-8"))
    assert data["status"] == "failed"
    calls = [c for c in data["invocations"] if c["component"] == branch]
    assert len(calls) == 1 and calls[0]["status"] == "failed"
    assert calls[0]["success_raw"] == "maybe"
    assert calls[0]["success_normalized"] is None
    assert calls[0]["success_normalization_status"] == "invalid"
    assert '"maybe"' in calls[0]["raw_output"]["content"]
    assert not any(c["component"] == "summary" for c in data["invocations"])


def test_missing_logprobs_no_fabricated_nll(tmp_path):
    _, trace, _, _ = run_synthetic(tmp_path, logprobs=False)
    assert all(c["logprobs_status"] == "unsupported" and c["nll"]["nll_mean"] is None for c in trace.data["invocations"])


def test_malformed_structured_output_retained(tmp_path):
    llm = SyntheticLLM("{not JSON", [], "extractor")
    with pytest.raises(json.JSONDecodeError):
        with trajectory_context("broken", LLMConfig(), enabled=True, output_root=tmp_path, metadata={"synthetic": True}):
            invoke_component("qa", ChatPromptTemplate.from_template("{question}"), {"question": "q"}, QAAnswerFormat, llm=llm)
    data = json.loads(next(tmp_path.rglob("trajectory.json")).read_text(encoding="utf-8"))
    assert data["invocations"][0]["raw_output"]["content"] == "{not JSON"
    assert data["invocations"][0]["status"] == "failed"


def test_unique_run_directories_no_overwrite(tmp_path):
    _, a, _, _ = run_synthetic(tmp_path)
    before = a.path.read_bytes()
    _, b, _, _ = run_synthetic(tmp_path)
    assert a.path != b.path and a.path.read_bytes() == before


def test_empty_retrieval_no_invented_extractions(tmp_path, monkeypatch):
    from agents.plan_executor import build_plan_executor
    factory = SyntheticFactory()
    monkeypatch.setattr("src.invocation.build_llm", factory)
    def empty(query):
        return [], []
    with trajectory_context("empty", LLMConfig(), enabled=True, output_root=tmp_path, metadata={"synthetic": True}) as trace:
        output = build_plan_executor(empty).invoke({"original_question": "q", "plan": ["find", "combine"], "stop": False})
    assert len(output["step_output"]) == 2
    assert not any(c["component"] == "extractor" for c in trace.data["invocations"])
    assert trace.data["steps"][0]["execution_masks"]["extractor"] is False
    retrieval = trace.data["retrievals"][0]
    assert retrieval["top_k_returned"] == 0 and retrieval["top1_uncertainty"] is None


@pytest.mark.parametrize("branch", ["qa", "aggregate"])
def test_missing_success_in_both_branches(tmp_path, monkeypatch, branch):
    from agents.plan_executor import build_plan_executor
    factory = SyntheticFactory()
    factory.queues[branch][0].pop("success")
    monkeypatch.setattr("src.invocation.build_llm", factory)
    with pytest.raises(SuccessValidationError):
        with trajectory_context("missing", LLMConfig(), enabled=True, output_root=tmp_path, metadata={"synthetic": True}):
            build_plan_executor(SyntheticRetriever()).invoke({"original_question": "q", "plan": ["find", "combine"], "stop": False})
    data = json.loads(next(tmp_path.rglob("trajectory.json")).read_text(encoding="utf-8"))
    call = data["invocations"][-1]
    assert call["component"] == branch and call["success_normalization_status"] == "missing"
    assert call["success_raw"] is None and call["success_normalized"] is None


def test_retrieval_error_is_persisted_and_not_retried(tmp_path, monkeypatch):
    from agents.plan_executor import build_plan_executor
    factory = SyntheticFactory()
    monkeypatch.setattr("src.invocation.build_llm", factory)
    attempts = []
    def broken(query):
        attempts.append(query)
        raise RuntimeError("Synthetic retriever error")
    with pytest.raises(RuntimeError):
        with trajectory_context("retrieval-failure", LLMConfig(), enabled=True, output_root=tmp_path, metadata={"synthetic": True}):
            build_plan_executor(broken).invoke({"original_question": "q", "plan": ["find"], "stop": False})
    data = json.loads(next(tmp_path.rglob("trajectory.json")).read_text(encoding="utf-8"))
    assert len(attempts) == 1 and data["status"] == "failed"
    assert data["retrievals"][0]["status"] == "failed"
    assert data["retrievals"][0]["step_id"] == "step:0"
    assert data["retrievals"][0]["raw_faiss_scores"] is None


def test_nonfinite_provider_values_persist_with_explicit_type(tmp_path):
    from src.telemetry import atomic_json
    path = tmp_path / "nonfinite.json"
    atomic_json(path, {"raw": [float("nan"), float("inf"), -float("inf")]})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert [v["value"] for v in data["raw"]] == ["nan", "inf", "-inf"]
    assert all(v["serialization_type"] == "nonfinite_float" for v in data["raw"])
