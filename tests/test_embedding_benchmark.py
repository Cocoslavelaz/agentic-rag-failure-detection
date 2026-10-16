from dataclasses import replace
import numpy as np
import pytest
from src.embedding_benchmark import encode_with_backoff, compare_embeddings
from src.embedding_encoder import EncoderOOMError
from src.embedding_contract import EncoderConfig


def test_oom_backoff_reencodes_every_original_document_and_restores_config():
    class Encoder:
        config = EncoderConfig()
        last_stats = {'synthetic': True}
        calls = []
        def encode_documents(self, texts):
            self.calls.append((self.config.batch_size, list(texts)))
            if self.config.batch_size > 8:
                raise EncoderOOMError('synthetic CUDA OOM')
            return np.eye(768,dtype=np.float32)[:len(texts)]
    encoder = Encoder()
    original = encoder.config
    vectors, report = encode_with_backoff(encoder, ['first','second','third'], 32)
    assert encoder.calls == [(32,['first','second','third']), (16,['first','second','third']), (8,['first','second','third'])]
    assert len(vectors) == 3 and report['effective_batch_size'] == 8
    assert [a['status'] for a in report['attempts']] == ['cuda_oom','cuda_oom','success']
    assert encoder.config == original


def test_consistency_accepts_small_fp32_differences_and_rejects_wrong_rows():
    a = np.eye(768,dtype=np.float32)[:3]
    b = a.copy()
    b[:,7] += 1e-6
    assert compare_embeddings(a,b)['passed']
    with pytest.raises(ValueError,match='consistency failed'):
        compare_embeddings(a,b[::-1])
    with pytest.raises(ValueError):
        compare_embeddings(a,b[:1])


def test_failed_all_batch_sizes_keeps_configuration_and_raises():
    class Encoder:
        config = EncoderConfig()
        def encode_documents(self,texts):
            raise EncoderOOMError('synthetic failure')
    encoder = Encoder()
    original = encoder.config
    with pytest.raises(EncoderOOMError,match='All benchmark batch sizes failed'):
        encode_with_backoff(encoder,['text'],8)
    assert encoder.config == original
