"""Bounded benchmark helpers; OOM retries encode the entire identical input."""
from dataclasses import asdict, replace
import time
import numpy as np
from src.embedding_encoder import EncoderOOMError


def encode_with_backoff(encoder, texts, requested_batch, *, before_attempt=lambda: None,
                        after_attempt=lambda: {}, clear_oom=lambda: None):
    if not 1 <= len(texts) <= 5000 or requested_batch not in (8, 16, 32):
        raise ValueError('Bounded benchmark requires 1..5000 texts and batch 8/16/32')
    original = encoder.config
    attempts = []
    batch = requested_batch
    try:
        while batch >= 1:
            encoder.config = replace(original, batch_size=batch)
            before_attempt()
            started = time.perf_counter()
            try:
                vectors = encoder.encode_documents(texts)
            except EncoderOOMError as exc:
                attempts.append({'batch_size': batch, 'status': 'cuda_oom', 'message': str(exc),
                                 'seconds': time.perf_counter()-started, **after_attempt()})
                clear_oom()
                batch //= 2
                continue
            metrics = after_attempt()
            elapsed = time.perf_counter()-started
            if len(vectors) != len(texts):
                raise ValueError('Benchmark encoder lost rows')
            attempts.append({'batch_size': batch, 'status': 'success', 'seconds': elapsed, **metrics})
            return vectors, {'requested_batch_size': requested_batch, 'effective_batch_size': batch,
                'encoder_configuration': asdict(encoder.config), 'encoder_configuration_hash': encoder.config.config_hash,
                'attempts': attempts, 'seconds': elapsed, 'documents_per_second': len(texts)/elapsed,
                'document_count': len(texts), 'encoding_statistics': encoder.last_stats, **metrics}
        raise EncoderOOMError('All benchmark batch sizes failed; no documents accepted or skipped')
    finally:
        encoder.config = original


def compare_embeddings(cpu, gpu, *, max_absolute_tolerance=1e-4, min_cosine=0.99999):
    if cpu.shape != gpu.shape or cpu.dtype != np.float32 or gpu.dtype != np.float32 or not len(cpu):
        raise ValueError('CPU/GPU embedding shape/dtype mismatch')
    a, b = cpu.astype(np.float64), gpu.astype(np.float64)
    cosine = np.sum(a*b, axis=1)/(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1))
    difference = np.abs(a-b)
    maximum, minimum = float(difference.max()), float(cosine.min())
    passed = bool(np.isfinite(difference).all() and np.isfinite(cosine).all()
                  and maximum <= max_absolute_tolerance and minimum >= min_cosine)
    result = {'passed': passed, 'row_count': len(a), 'maximum_absolute_difference': maximum,
        'mean_absolute_difference': float(difference.mean()), 'minimum_per_document_cosine': minimum,
        'mean_per_document_cosine': float(cosine.mean()), 'maximum_l2_difference': float(np.linalg.norm(a-b, axis=1).max()),
        'max_absolute_tolerance': max_absolute_tolerance, 'minimum_cosine_required': min_cosine}
    if not passed:
        raise ValueError(f'CPU/GPU numerical consistency failed: {result}')
    return result
