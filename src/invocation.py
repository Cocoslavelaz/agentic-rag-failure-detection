"""Pass identical prompts through one backend and retain raw parse failures."""
import os
import json
from time import perf_counter

from .backend import LLMConfig, TEMPERATURES, build_llm
from .success import MISSING, normalize_success, validate_qa_output
from .telemetry import ACTIVE_TRACE, STEP_CONTEXT, digest, now
from .uncertainty import generated_logprobs, nll_from_logprobs


def invoke_component(component, prompt, inputs, schema=None, *, step_id=None,
                     document_id=None, retrieval_rank=None, llm=None):
    config = LLMConfig.from_env()
    messages = prompt.invoke(inputs).to_messages()
    trace = ACTIVE_TRACE.get()
    context = STEP_CONTEXT.get()
    event = {"component": component, "step_id": step_id if step_id is not None else context.get("step_id"),
             "document_id": document_id, "retrieval_rank": retrieval_rank,
             "route": context.get("route", "summary" if component == "summary" else "plan" if component == "planner" else "define"),
             "model_id": config.model, "temperature": TEMPERATURES[component],
             "thinking_mode": config.thinking if config.backend == "ollama" else None,
             "prompt_hash": digest([{"role": m.type, "content": m.content} for m in messages]),
             "input": inputs, "raw_output": None, "parsed_output": None,
             "token_usage": None, "token_logprobs": None, "logprobs_status": "not_observed",
             "nll": None, "latency_ms": None, "status": "running", "error": None,
             "started_at": now()}
    if trace:
        event = trace.begin_call(event)
    start = perf_counter()
    try:
        model = llm or build_llm(component, config)
        kwargs = {"stream": False, "logprobs": os.getenv("LLM_LOGPROBS", "true").lower() == "true"} if config.backend == "ollama" else {}
        parsing_error = None
        if schema is None:
            raw = model.invoke(messages, **kwargs)
            payload = raw.content
        elif config.backend == "ollama":
            # include_raw's RunnableMap drops invoke kwargs in this pinned
            # LangChain version. Bind the native format at the actual model call
            # so stream=False and logprobs=True reach Ollama, with no second call.
            raw = model.invoke(messages, format=schema.model_json_schema(), **kwargs)
            payload = None
        else:
            # The exact original wire schema is used. Dict transport lets us
            # observe boolean/integer success before the original str parser.
            options = {"method": "json_schema"} if config.backend == "ollama" else {}
            result = model.with_structured_output(schema.model_json_schema(), include_raw=True, **options).invoke(messages, **kwargs)
            raw = result["raw"]
            event["raw_output"] = raw.model_dump(mode="json")
            parsing_error = result.get("parsing_error")
            payload = result["parsed"]
        event["raw_output"] = raw.model_dump(mode="json")
        event["token_usage"] = raw.usage_metadata or {k: raw.response_metadata.get(k) for k in ("prompt_eval_count", "eval_count")}
        values, status = generated_logprobs(raw.response_metadata)
        if config.backend == "openai":
            # Do not assert the optional OpenAI backend lacks logprobs.
            status = "not_requested" if values is None else status
        elif not kwargs["logprobs"] and values is None:
            status = "not_requested"
        event["token_logprobs"] = event["raw_output"]["response_metadata"].get("logprobs")
        event["logprobs_status"] = status
        event["nll"] = nll_from_logprobs(values, status=status)
        event["nll"]["reported_output_token_count"] = raw.response_metadata.get("eval_count")
        event["nll"]["token_count_matches_usage"] = (len(values) == raw.response_metadata["eval_count"]
            if values is not None and raw.response_metadata.get("eval_count") is not None else None)
        if parsing_error is not None:
            raise parsing_error
        if schema is not None and config.backend == "ollama":
            payload = json.loads(raw.content)
        if component in {"qa", "aggregate"}:
            norm = normalize_success(payload.get("success", MISSING) if isinstance(payload, dict) else MISSING)
            event.update(norm.fields())
            parsed = validate_qa_output(payload, schema)
            event["parsed_output"] = parsed
        elif schema is not None:
            parsed = schema.model_validate(payload)
            event["parsed_output"] = parsed.model_dump(mode="json")
        else:
            parsed = payload
            event["parsed_output"] = parsed
        event["status"] = "completed"
        return parsed
    except Exception as exc:
        event.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        event.update(latency_ms=(perf_counter()-start)*1000, finished_at=now())
        if trace:
            trace.save()
