"""Utilities consume observed scores only; no inferred token probabilities."""
import math
import statistics


def nll_from_logprobs(values, *, status="available", scope="full_response"):
    result = dict(status=status, scope=scope, nll_sum=None, nll_mean=None,
                  nll_max=None, nll_std=None, token_count=None)
    if status in {"unsupported", "not_requested", "returned_unverified"}:
        return result
    if values is None:
        return dict(result, status="missing")
    values = list(values)
    if not values:
        return dict(result, status="empty", token_count=0)
    if any(type(x) not in (float, int) or not math.isfinite(x) or x > 0 for x in values):
        return dict(result, status="invalid")
    losses = [-float(x) for x in values]
    return dict(result, status="available", nll_sum=math.fsum(losses),
                nll_mean=statistics.mean(losses), nll_max=max(losses),
                nll_std=statistics.pstdev(losses), token_count=len(losses))


def retrieval_features(scores, *, cosine_verified=False):
    scores = list(scores)
    result = dict(top1_uncertainty=None, topk_mean_uncertainty=None, score_margin=None,
                  similarity_validity_status="cosine_verified" if cosine_verified else "unverified_inner_product")
    if not scores:
        return dict(result, similarity_validity_status="empty")
    if any(not math.isfinite(x) for x in scores):
        return dict(result, similarity_validity_status="invalid_scores")
    # A ranking-gap feature is valid for raw IP too; it is never a probability.
    if len(scores) >= 2:
        result["score_margin"] = scores[0] - scores[1]
    if cosine_verified:
        if any(x < -1.00001 or x > 1.00001 for x in scores):
            return dict(result, similarity_validity_status="invalid_cosine_range")
        result.update(top1_uncertainty=1-scores[0],
                      topk_mean_uncertainty=1-statistics.mean(scores))
    return result


def generated_logprobs(raw):
    """Accept only identifiable generated-token records; preserve other shapes."""
    records = raw.get("logprobs") if isinstance(raw, dict) else None
    if isinstance(records, dict):
        records = records.get("content")
    if records is None:
        return None, "unsupported"
    if isinstance(records, list):
        # Ollama SDK returns Logprob Pydantic objects; raw HTTP returns dicts.
        records = [r.model_dump() if hasattr(r, "model_dump") else r for r in records]
    if not isinstance(records, list) or any(not isinstance(r, dict) or "logprob" not in r or "token" not in r for r in records):
        return None, "returned_unverified"
    return [r["logprob"] for r in records], "available"


def logprobs_coverage(raw, content=None):
    values, availability = generated_logprobs(raw)
    count = len(values) if values is not None else None
    evaluated = raw.get("eval_count")
    ratio = count/evaluated if count is not None and type(evaluated) is int and evaluated > 0 else None
    if values is None:
        status = availability
    elif type(evaluated) is not int or evaluated < 0:
        status = "unknown"
    elif evaluated == 0:
        status = "empty" if count == 0 else "inconsistent"
    else:
        status = "complete" if count == evaluated else "partial" if count < evaluated else "inconsistent"
    records = raw.get("logprobs")
    if isinstance(records, dict):
        records = records.get("content")
    matches = None
    if values is not None and content is not None:
        records = [r.model_dump() if hasattr(r, "model_dump") else r for r in records]
        if all(isinstance(r.get("token"), str) for r in records):
            matches = "".join(r["token"] for r in records) == content
    return {"logprobs_count": count, "eval_count": evaluated, "coverage_ratio": ratio,
            "coverage_status": status, "logprob_text_matches_content": matches,
            "missing_logprobs_count": evaluated-count if status == "partial" else None,
            "done_reason": raw.get("done_reason")}
