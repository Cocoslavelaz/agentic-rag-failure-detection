import math
import pytest
from src.uncertainty import generated_logprobs, nll_from_logprobs, retrieval_features


def test_nll_numeric_correctness():
    r = nll_from_logprobs([-1, -2, -3])
    assert r["nll_sum"] == 6
    assert r["nll_mean"] == 2
    assert r["nll_max"] == 3
    assert r["nll_std"] == pytest.approx(math.sqrt(2/3))
    assert r["token_count"] == 3


@pytest.mark.parametrize("values,status", [(None, "missing"), ([], "empty"), ([None], "invalid"),
    ([float("nan")], "invalid"), ([float("inf")], "invalid"), ([-float("inf")], "invalid"),
    ([1], "invalid"), ([True], "invalid"), (["-1"], "invalid")])
def test_nll_missing_and_invalid(values, status):
    r = nll_from_logprobs(values)
    assert r["status"] == status and r["nll_mean"] is None


def test_unsupported_and_unknown_logprob_format():
    assert nll_from_logprobs(None, status="unsupported")["status"] == "unsupported"
    assert generated_logprobs({"eval_count": 10}) == (None, "unsupported")
    assert generated_logprobs({"logprobs": [0.7]}) == (None, "returned_unverified")
    assert generated_logprobs({"logprobs": [{"token": "a", "logprob": -0.5}]}) == ([-0.5], "available")
    from ollama import ChatResponse
    response = ChatResponse(message={"role": "assistant", "content": "a"}, logprobs=[{"token": "a", "logprob": -0.5}])
    assert generated_logprobs({"logprobs": response.logprobs}) == ([-0.5], "available")


def test_verified_cosine_features_and_ranking_preserved():
    scores = [0.9, 0.8, 0.4]
    before = scores.copy()
    r = retrieval_features(scores, cosine_verified=True)
    assert scores == before
    assert r["top1_uncertainty"] == pytest.approx(0.1)
    assert r["topk_mean_uncertainty"] == pytest.approx(0.3)
    assert r["score_margin"] == pytest.approx(0.1)


def test_non_normalized_embeddings_not_cosine():
    r = retrieval_features([3, 2], cosine_verified=False)
    assert r["top1_uncertainty"] is None and r["topk_mean_uncertainty"] is None
    assert r["score_margin"] == 1


def test_empty_and_single_retrieval():
    assert retrieval_features([], cosine_verified=True)["similarity_validity_status"] == "empty"
    assert retrieval_features([0.4], cosine_verified=True)["score_margin"] is None
    assert retrieval_features([0.4], cosine_verified=True)["top1_uncertainty"] == pytest.approx(0.6)
    assert retrieval_features([2], cosine_verified=True)["similarity_validity_status"] == "invalid_cosine_range"
    assert retrieval_features([float("nan")])["similarity_validity_status"] == "invalid_scores"
