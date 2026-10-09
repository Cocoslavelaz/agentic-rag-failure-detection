import math
import pickle
from types import SimpleNamespace
import faiss
import numpy as np
import pytest
import torch
from corpus.retrieve import Retriever
from src.backend import LLMConfig
from src.telemetry import trajectory_context
from src.utils import RetrieveTopChunk


class Embedding:
    device = "cpu"
    config = SimpleNamespace(_name_or_path="synthetic-unit-vector-model")

    def __call__(self, **kwargs):
        return SimpleNamespace(last_hidden_state=torch.tensor([[[3.0, 0.0]]]))


class Corpus:
    def get(self, doc_id):
        return SimpleNamespace(text=f"text {doc_id}", doc_id=doc_id)


@pytest.mark.parametrize("normalized", [True, False])
def test_index_norm_validation_actual_faiss_and_ranking(tmp_path, monkeypatch, normalized):
    docs = np.array([[0.8, 0.6], [0.9, math.sqrt(0.19)]], dtype="float32")
    if not normalized:
        docs *= 3
    partition = tmp_path / "dpr100_0"
    with partition.open("wb") as handle:
        pickle.dump((["doc-a", "doc-b"], docs), handle)
    retriever = Retriever(gpu_ids=[])
    monkeypatch.setattr(retriever, "_move_index_to_gpu", lambda: None)
    monkeypatch.setenv("TRAJECTORY_ENABLED", "true")
    retriever.init_index_and_add(str(tmp_path), "dpr100")
    assert retriever.document_norms_verified == normalized
    assert retriever.index_version is not None
    tool = RetrieveTopChunk(tokenizer=lambda *a, **kw: {}, embedding_model=Embedding(),
                            retrieval_model=retriever, corpus=Corpus(), top_k=2)
    without_logging = tool("q")
    with trajectory_context("retriever-test", LLMConfig(), enabled=True, output_root=tmp_path / "logs", metadata={"synthetic": True}) as trace:
        with_logging = tool("q")
    assert with_logging == without_logging
    assert with_logging[1] == ["doc-b", "doc-a"]
    event = trace.data["retrievals"][0]
    assert event["ordered_document_ids"] == ["doc-b", "doc-a"]
    assert event["raw_faiss_scores"] == pytest.approx([0.9, 0.8] if normalized else [2.7, 2.4])
    assert event["query_norms_verified"] is True
    if normalized:
        assert event["top1_uncertainty"] == pytest.approx(0.1)
        assert event["topk_mean_uncertainty"] == pytest.approx(0.15)
        assert event["similarity_validity_status"] == "cosine_verified"
    else:
        assert event["cosine_similarities"] is None
        assert event["top1_uncertainty"] is None


def test_search_retains_full_raw_faiss_hits(tmp_path):
    retriever = Retriever(gpu_ids=[])
    retriever.index = faiss.IndexFlatIP(2)
    retriever.index.add(np.array([[1.0, 0.0], [0.8, 0.6]], dtype="float32"))
    retriever.doc_lookup = ["a", "b"]
    result = retriever.search([0], np.array([[1.0, 0.0]], dtype="float32"), top_k=2)
    assert list(result[0]) == ["a", "b"]
    assert [h["document_id"] for h in retriever.last_search["faiss_hits"][0]] == ["a", "b"]
    assert all(h["valid_index_position"] for h in retriever.last_search["faiss_hits"][0])
