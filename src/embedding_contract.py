"""Lightweight shared vector contract; imports neither Torch nor FAISS."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import numpy as np

MODEL_REVISION = '9bbca17d9273fd0d03d5725c7a4b0f6b45142062'
CODE_REVISION = '40ced75c3017eb27626c9d4ea981bde21a2662f4'


def stable_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def sha256_file(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


@dataclass(frozen=True)
class EncoderConfig:
    model_id: str = 'Alibaba-NLP/gte-multilingual-base'
    model_revision: str = MODEL_REVISION
    tokenizer_revision: str = MODEL_REVISION
    code_revision: str = CODE_REVISION
    pooling: str = 'cls_last_hidden_state'
    dimension: int = 768
    document_max_length: int = 512
    query_max_length: int = 512
    normalize: bool = True
    model_dtype: str = 'float32'
    output_dtype: str = 'float32'
    attention_implementation: str = 'sdpa'
    batch_size: int = 8
    seed: int = 42
    device: str = 'cpu'
    cpu_threads: int = 4

    def __post_init__(self):
        if any(not re.fullmatch('[0-9a-f]{40}', revision) for revision in
               (self.model_revision, self.tokenizer_revision, self.code_revision)):
            raise ValueError('Model, tokenizer and code revisions must be full commit SHAs')
        if self.pooling != 'cls_last_hidden_state' or self.dimension != 768 or not self.normalize:
            raise ValueError('Adaptation requires CLS 768-d and L2 normalization; no implicit dimension slicing')
        if self.model_dtype != 'float32' or self.output_dtype != 'float32':
            raise ValueError('Phase 3A configuration supports explicit float32 only')
        if not (2 <= self.document_max_length <= 8192 and 2 <= self.query_max_length <= 8192):
            raise ValueError('Token limits must be between 2 and model maximum 8192')
        if self.batch_size < 1 or self.cpu_threads < 1:
            raise ValueError('Batch size and CPU threads must be positive')
        if self.device not in {'cpu', 'cuda', 'auto'}:
            raise ValueError('device must be cpu, cuda or explicit auto fallback policy')

    @property
    def config_hash(self):
        return hashlib.sha256(stable_json(asdict(self)).encode('utf-8')).hexdigest()


def validate_vectors(vectors, dimension, *, tolerance=1e-5):
    if vectors.dtype != np.float32 or vectors.ndim != 2 or vectors.shape[1] != dimension:
        raise ValueError('Embeddings must be an aligned float32 matrix with the declared dimension')
    if not np.all(np.isfinite(vectors)):
        raise ValueError('Non-finite embedding values')
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if len(norms) and not np.allclose(norms, 1, atol=tolerance, rtol=0):
        raise ValueError('Embedding L2 norms are not unit length')
    return {'count': len(norms), 'finite': True,
            'norm_min': float(norms.min()) if len(norms) else None,
            'norm_max': float(norms.max()) if len(norms) else None}


