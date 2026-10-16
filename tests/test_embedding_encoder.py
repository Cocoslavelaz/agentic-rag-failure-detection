from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from src.embedding_encoder import EncoderConfig, EncoderOOMError, GTEEncoder, validate_vectors
import torch


class FakeTokenizer:
    cls_token_id = 0
    def __call__(self, texts, *, truncation=False, max_length=None, return_tensors=None, **kwargs):
        rows = [[0] + [3 + len(word) for word in text.split()] + [2] for text in texts]
        if truncation:
            rows = [row[:max_length - 1] + [2] if len(row) > max_length else row for row in rows]
        if return_tensors is None:
            return {'input_ids': rows}
        length = max(map(len, rows))
        return {'input_ids': torch.tensor([row + [1] * (length - len(row)) for row in rows]),
                'attention_mask': torch.tensor([[1] * len(row) + [0] * (length - len(row)) for row in rows])}


class FakeModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        assert torch.is_inference_mode_enabled()
        hidden = torch.full((*input_ids.shape, 768), 10000.0)
        hidden[:, 0, :] = torch.arange(1, 769).float() + input_ids[:, 1:2]
        return SimpleNamespace(last_hidden_state=hidden)


def encoder(config=None, model=None):
    return GTEEncoder(config or EncoderConfig(), FakeTokenizer(), model or FakeModel())


def test_cls_pooling_shape_l2_and_finite():
    result = encoder().encode_documents(['one two', 'three'])
    assert result.shape == (2, 768) and result.dtype == np.float32
    expected = np.arange(1, 769, dtype=np.float32) + 6
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(result[0], expected, atol=1e-7)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6)


def test_document_query_consistency_batching_and_eval():
    instance = encoder(replace(EncoderConfig(), batch_size=1))
    docs = instance.encode_documents(['one two', 'three'])
    queries = instance.encode_queries(['one two', 'three'])
    np.testing.assert_array_equal(docs, queries)
    assert not instance.model.training
    assert len(instance.last_stats['batch_inference_seconds']) == 2


def test_empty_input_and_long_text_truncation():
    instance = encoder(replace(EncoderConfig(), document_max_length=8))
    assert instance.encode_documents([]).shape == (0, 768)
    instance.encode_documents(['word ' * 20, 'short'])
    assert instance.last_stats['untruncated_token_lengths'] == [22, 3]
    assert instance.last_stats['effective_token_lengths'] == [8, 3]
    assert instance.last_stats['truncated_count'] == 1


def test_explicit_cpu_fallback_and_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    instance = encoder(replace(EncoderConfig(), device='auto'))
    assert instance.device == 'cpu' and instance.device_resolution['fallback_reason'] == 'CUDA unavailable'
    with pytest.raises(RuntimeError, match='CUDA explicitly requested'):
        encoder(replace(EncoderConfig(), device='cuda'))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 0.0])
def test_invalid_cls_values_rejected(value):
    class Broken(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.full((*input_ids.shape, 768), value))
    with pytest.raises(ValueError, match='Invalid or zero CLS'):
        encoder(model=Broken()).encode_documents(['text'])


def test_oom_is_explicit_and_never_skips_batch():
    class OOM(torch.nn.Module):
        def forward(self, **kwargs):
            raise torch.cuda.OutOfMemoryError('synthetic OOM')
    with pytest.raises(EncoderOOMError, match='no documents skipped'):
        encoder(model=OOM()).encode_documents(['text'])


@pytest.mark.parametrize('kwargs', [{'model_revision': 'main'}, {'dimension': 128}, {'normalize': False}, {'batch_size': 0}])
def test_configuration_constraints(kwargs):
    with pytest.raises(ValueError):
        EncoderConfig(**kwargs)


def test_vectors_must_be_fp32_unit_and_finite():
    for matrix in [np.ones((2, 768), np.float32), np.zeros((2, 768), np.float32),
                   np.full((2, 768), np.nan, np.float32), np.ones((2, 768), np.float64)]:
        with pytest.raises(ValueError):
            validate_vectors(matrix, 768)


def test_local_loading_refuses_model_revision_before_import(tmp_path):
    import json
    config = EncoderConfig()
    lock = tmp_path / 'lock.json'
    lock.write_text(json.dumps({'model_id': config.model_id, 'model_revision': 'f' * 40}), encoding='utf-8')
    with pytest.raises(ValueError, match='lock mismatch'):
        GTEEncoder.from_local(tmp_path, lock, config)


def test_empty_string_not_encoded():
    with pytest.raises(ValueError, match='nonempty'):
        encoder().encode_documents([''])
