"""Optional, per-question crash-safe sidecar; agent state/output stays separate."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import uuid

ACTIVE_TRACE = ContextVar("ma_rag_trace", default=None)
STEP_CONTEXT = ContextVar("ma_rag_step", default={})
ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(data), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def json_safe(value):
    # JSON has no NaN/Infinity literals. Preserve their identity explicitly
    # rather than dropping a malformed provider response or inventing a score.
    if isinstance(value, float) and not math.isfinite(value):
        return {"serialization_type": "nonfinite_float", "value": str(value)}
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def baseline_metadata(config):
    from .backend import TEMPERATURES
    from . import prompt_template
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    metadata = {"configuration_id": config.configuration_id, "backend": config.backend,
                "model_id": config.model, "thinking_mode": config.thinking if config.backend == "ollama" else None,
                "context_length_requested": config.num_ctx if config.backend == "ollama" else None,
                "temperatures": TEMPERATURES, "git_commit_sha": sha.stdout.strip() or None,
                "git_dirty": bool(dirty.stdout.strip()), "structured_output_mode": "json_schema" if config.backend == "ollama" else "provider_default",
                "prompt_hashes": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in vars(prompt_template).items()
                                  if isinstance(v, str) and ("message" in k)},
                "implementation_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for directory in ("agents", "src", "corpus") for p in sorted((ROOT/directory).glob("*.py"))},
                "retriever": {"embedding_model": "hf/gte-multilingual-base", "top_k": 10,
                              "index_type": "IndexFlatIP", "index_version": "unverified_until_retrieval"}}
    metadata["packages"] = {}
    for package in ("langchain-core", "langchain-ollama", "langchain-openai", "ollama", "pydantic", "langgraph"):
        try:
            metadata["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metadata["packages"][package] = None
    if config.backend == "ollama":
        from .ollama_api import request
        try:
            tags = request(config.base_url, "/api/tags")
            tag = next((t for t in tags["models"] if t["name"] == config.model), None)
            if tag is None:
                raise ValueError("Exact configured model not installed; no download/fallback")
            metadata.update(model_tag=tag["name"], model_digest=tag["digest"],
                            quantization=tag.get("details", {}).get("quantization_level"),
                            ollama_version=request(config.base_url, "/api/version")["version"])
            show = request(config.base_url, "/api/show", {"model": config.model})
            metadata["model_capabilities"] = show.get("capabilities")
            metadata["model_context_capacity"] = {k: v for k, v in show.get("model_info", {}).items() if "context_length" in k}
            metadata["model_template_sha256"] = hashlib.sha256(show.get("template", "").encode()).hexdigest()
            metadata["model_parameters"] = show.get("parameters")
        except Exception as exc:
            metadata["metadata_error"] = f"{type(exc).__name__}: {exc}"
    return metadata


class Trajectory:
    def __init__(self, question_id, config, *, output_root=None, dataset_id=None, metadata=None):
        run_id = str(uuid.uuid4())
        self.data = {"schema_version": "ma_rag_trajectory_v1", "run_id": run_id,
                     "question_id": str(question_id), "dataset_id": dataset_id,
                     "timestamp": now(), "status": "running", "error": None,
                     "metadata": metadata if metadata is not None else baseline_metadata(config),
                     "invocations": [], "retrievals": [], "steps": []}
        self.directory = Path(output_root or os.getenv("TRAJECTORY_DIR", ROOT / "outputs" / "trajectories")) / config.configuration_id / run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "trajectory.json"
        self.save()

    def save(self):
        atomic_json(self.path, self.data)

    def begin_call(self, event):
        event = {"run_id": self.data["run_id"], "question_id": self.data["question_id"],
                 "invocation_id": f"{self.data['run_id']}:call:{len(self.data['invocations'])}",
                 **event}
        self.data["invocations"].append(event)
        self.save()
        return event

    def retrieval(self, event):
        context = STEP_CONTEXT.get()
        self.data["retrievals"].append({"step_id": context.get("step_id"), "route": "rag", **event})
        self.data["metadata"].setdefault("retriever", {}).update({k: event[k] for k in
            ("embedding_model", "index_type", "index_version", "document_norms_verified") if k in event})
        self.save()


@contextmanager
def trajectory_context(question_id, config=None, *, enabled=None, output_root=None, dataset_id=None, metadata=None):
    from .backend import LLMConfig
    if enabled is None:
        enabled = os.getenv("TRAJECTORY_ENABLED", "false").lower() == "true"
    trace = Trajectory(question_id, config or LLMConfig.from_env(), output_root=output_root,
                       dataset_id=dataset_id, metadata=metadata) if enabled else None
    token = ACTIVE_TRACE.set(trace)
    try:
        yield trace
        if trace:
            trace.data.update(status="completed", finished_at=now())
            if metadata is None and (config or LLMConfig.from_env()).backend == "ollama":
                from .ollama_api import request
                try:
                    trace.data["metadata"]["loaded_models_after_run"] = request((config or LLMConfig.from_env()).base_url, "/api/ps")
                except Exception as exc:
                    trace.data["metadata"]["runtime_metadata_error"] = str(exc)
    except Exception as exc:
        if trace:
            trace.data.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)}, finished_at=now())
        raise
    finally:
        if trace:
            trace.save()
        ACTIVE_TRACE.reset(token)


@contextmanager
def step_context(step_id, route):
    token = STEP_CONTEXT.set({"step_id": step_id, "route": route})
    trace = ACTIVE_TRACE.get()
    step = {"step_id": step_id, "route": route, "status": "running",
            "execution_masks": {"retriever": False, "extractor": False, "qa": False, "aggregate": False}}
    if trace:
        trace.data["steps"].append(step)
        trace.save()
    try:
        yield step
        step["status"] = "completed"
    except Exception as exc:
        step.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if trace:
            ids = [e["component"] for e in trace.data["invocations"] if e["step_id"] == step_id]
            step["execution_masks"] = {k: (any(r["step_id"] == step_id for r in trace.data["retrievals"]) if k == "retriever" else k in ids)
                                       for k in step["execution_masks"]}
            trace.save()
        STEP_CONTEXT.reset(token)
