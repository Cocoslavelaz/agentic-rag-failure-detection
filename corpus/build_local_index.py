"""Persistent CPU FlatIP engineering index. Phase 3A never builds >1,000 docs."""
from dataclasses import asdict
import hashlib
import heapq
import json
from pathlib import Path
import sys
import time

import faiss
import numpy as np
from importlib.metadata import version

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.embedding_contract import stable_json, sha256_file, validate_vectors


from src.corpus_selection import fingerprint, verify_corpus, select_smoke_documents


def local_path(value):
    path = Path(value).resolve()
    # Full corpus/config/selection hashes exceed legacy MAX_PATH on Windows.
    if sys.platform == 'win32' and not str(path).startswith('\\\\?\\'):
        return Path('\\\\?\\' + str(path))
    return path


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(stable_json(value) + '\n')


def build_index(output_dir, documents, embeddings, *, corpus_checksum, corpus_manifest_hash,
                encoder_config, selection, runtime=None):
    """No normalization or ranking changes here: validate then add original vectors."""
    output_dir = local_path(output_dir)
    if not 1 <= len(documents) <= 1000:
        raise ValueError('Phase 3A only authorizes smoke indexes of at most 1,000 documents')
    config = asdict(encoder_config) if hasattr(encoder_config, '__dataclass_fields__') else dict(encoder_config)
    ids = [doc.get('document_id') for doc in documents]
    if any(not isinstance(identifier, str) or not identifier for identifier in ids) or len(set(ids)) != len(ids):
        raise ValueError('Duplicate/missing document IDs')
    if len(embeddings) != len(ids):
        raise ValueError('Document/vector count mismatch')
    norm_info = validate_vectors(embeddings, config['dimension'])
    if selection.get('document_ids') != ids:
        raise ValueError('Selection IDs do not match FAISS row order')
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / 'build_status.json', {'status': 'building', 'scope': 'smoke'})
    started = time.perf_counter()
    index = faiss.IndexFlatIP(config['dimension'])
    index.add(np.ascontiguousarray(embeddings))
    # Windows FAISS uses narrow paths in some builds: serialize through Python I/O.
    with (output_dir / 'index.faiss').open('xb') as handle:
        handle.write(faiss.serialize_index(index).tobytes())
    with (output_dir / 'embeddings.npy').open('xb') as handle:
        np.save(handle, embeddings, allow_pickle=False)
    with (output_dir / 'document_mapping.jsonl').open('x', encoding='utf-8', newline='\n') as mapping, \
            (output_dir / 'selected_documents.jsonl').open('x', encoding='utf-8', newline='\n') as texts:
        for row_id, document in enumerate(documents):
            mapping.write(stable_json({'row_id': row_id, 'document_id': document['document_id']}) + '\n')
            texts.write(stable_json(document) + '\n')
    files = ('index.faiss', 'embeddings.npy', 'document_mapping.jsonl', 'selected_documents.jsonl')
    manifest = {'schema_version': 1, 'builder_version': '2wiki-local-flatip-v1',
                'scope': 'smoke', 'research_retrieval_index': False, 'build_status': 'complete',
                'corpus_checksum': corpus_checksum, 'corpus_manifest_hash': corpus_manifest_hash,
                'number_of_documents': len(ids), 'embedding_dimension': config['dimension'],
                'encoder_config': config, 'encoder_config_hash': fingerprint(config),
                'model_id': config.get('model_id'), 'model_commit_sha': config.get('model_revision'),
                'tokenizer_revision': config.get('tokenizer_revision'), 'code_revision': config.get('code_revision'),
                'pooling': config.get('pooling'), 'document_max_length': config.get('document_max_length'),
                'query_max_length': config.get('query_max_length'), 'normalization': 'L2', 'embedding_dtype': 'float32',
                'faiss_index_type': 'IndexFlatIP', 'faiss_version': faiss.__version__,
                'torch_version': version('torch'), 'transformers_version': version('transformers'),
                'batch_size': config.get('batch_size'), 'device': (runtime or {}).get('resolved_device', config.get('device')),
                'random_seed': config.get('seed'), 'selection': selection, 'norm_validation': norm_info,
                'file_checksums': {name: sha256_file(output_dir / name) for name in files},
                'file_sizes_bytes': {name: (output_dir / name).stat().st_size for name in files},
                'index_checksum': sha256_file(output_dir / 'index.faiss'),
                'document_mapping_checksum': sha256_file(output_dir / 'document_mapping.jsonl')}
    write_json(output_dir / 'index_manifest.json', manifest)
    write_json(output_dir / 'build_complete.json', {'status': 'complete', 'index_manifest_sha256': sha256_file(output_dir / 'index_manifest.json')})
    return manifest, time.perf_counter() - started


def compare_cosine(index, queries, *, k=10, tolerance=1e-5):
    if type(k) is not int or k < 1:
        raise ValueError('k must be a positive integer')
    validate_vectors(queries, index.d)
    vectors = index.reconstruct_n(0, index.ntotal).astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    if not len(norms) or not np.all(np.isfinite(vectors)) or not np.allclose(norms, 1, atol=tolerance, rtol=0):
        raise ValueError('Index documents are not verified unit vectors')
    scores, rows = index.search(np.ascontiguousarray(queries), k)
    comparisons, strict_matches, tolerant_matches = [], [], []
    for query, found_scores, found_rows in zip(queries, scores, rows):
        q = query.astype(np.float64)
        cosine = vectors @ q / (norms * np.linalg.norm(q))
        independent_rows = np.argsort(-cosine, kind='stable')[:min(k, index.ntotal)]
        actual_rows = [int(row) for row in found_rows if row >= 0]
        strict_matches.append(actual_rows == independent_rows.tolist())
        tolerant_matches.append(bool(np.allclose(cosine[actual_rows], cosine[independent_rows], atol=tolerance, rtol=0)))
        for score, row in zip(found_scores, found_rows):
            if row < 0:
                continue
            comparisons.append({'row_id': int(row), 'faiss_raw_score': float(score),
                                'numpy_float64_cosine': float(cosine[row]),
                                'absolute_error': abs(float(score) - float(cosine[row]))})
    max_error = max((c['absolute_error'] for c in comparisons), default=None)
    verified = max_error is not None and max_error <= tolerance and all(tolerant_matches)
    if not verified:
        raise ValueError(f'FAISS/cosine validation failed; max_error={max_error}')
    return {'verified': True, 'max_absolute_error': max_error, 'tolerance': tolerance,
            'topk_exact_row_order_match': strict_matches,
            'topk_consistent_within_tolerance': tolerant_matches,
            'tie_policy': 'do not reorder FAISS; equal cosine ties may have different row ordering',
            'comparisons': comparisons}


class LocalFaissIndex:
    def compare_cosine(self, queries, *, k=10):
        return compare_cosine(self.index, queries, k=k)

    @classmethod
    def load(cls, directory, *, expected_corpus_checksum, expected_corpus_manifest_hash, expected_encoder_config):
        directory = local_path(directory)
        manifest_path = directory / 'index_manifest.json'
        complete = json.loads((directory / 'build_complete.json').read_text(encoding='utf-8'))
        if complete.get('status') != 'complete' or complete['index_manifest_sha256'] != sha256_file(manifest_path):
            raise ValueError('Incomplete or modified index manifest')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        config = asdict(expected_encoder_config) if hasattr(expected_encoder_config, '__dataclass_fields__') else expected_encoder_config
        if manifest.get('schema_version') != 1 or manifest['scope'] != 'smoke' or manifest['build_status'] != 'complete':
            raise ValueError('Wrong index scope/schema/status')
        if manifest['corpus_checksum'] != expected_corpus_checksum or manifest['corpus_manifest_hash'] != expected_corpus_manifest_hash:
            raise ValueError('Corpus checksum/manifest hash mismatch')
        if manifest['encoder_config'] != config or manifest['encoder_config_hash'] != fingerprint(config):
            raise ValueError('Encoder config/model revision mismatch')
        for name, checksum in manifest['file_checksums'].items():
            if sha256_file(directory / name) != checksum:
                raise ValueError(f'Index asset checksum mismatch: {name}')
        result = cls()
        # Python handles Chinese Windows paths; FAISS only sees bytes in memory.
        result.index = faiss.deserialize_index(np.frombuffer((directory / 'index.faiss').read_bytes(), dtype=np.uint8).copy())
        result.mapping = [json.loads(line) for line in (directory / 'document_mapping.jsonl').read_text(encoding='utf-8').splitlines()]
        result.documents = [json.loads(line) for line in (directory / 'selected_documents.jsonl').read_text(encoding='utf-8').splitlines()]
        ids = [row.get('document_id') for row in result.mapping]
        if (result.index.ntotal != manifest['number_of_documents'] or len(ids) != result.index.ntotal
                or len(result.documents) != result.index.ntotal or result.index.d != manifest['embedding_dimension']):
            raise ValueError('Index ntotal/dimension/mapping count mismatch')
        if (result.index.metric_type != faiss.METRIC_INNER_PRODUCT or not isinstance(result.index, faiss.IndexFlatIP)
                or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids)):
            raise ValueError('Invalid index type or duplicate/missing document IDs')
        if [row['row_id'] for row in result.mapping] != list(range(len(ids))) or [doc['document_id'] for doc in result.documents] != ids:
            raise ValueError('Row ID/document alignment mismatch')
        embeddings = np.load(directory / 'embeddings.npy', allow_pickle=False, mmap_mode='r')
        validate_vectors(embeddings, result.index.d)
        reconstructed = result.index.reconstruct_n(0, result.index.ntotal)
        if embeddings.shape != reconstructed.shape or not np.array_equal(embeddings, reconstructed):
            raise ValueError('Persisted embeddings and FAISS row vectors differ')
        result.manifest = manifest
        return result

    def search(self, queries, *, k=10):
        if type(k) is not int or k < 1:
            raise ValueError('k must be a positive integer')
        validate_vectors(queries, self.index.d)
        scores, rows = self.index.search(np.ascontiguousarray(queries), k)
        results = []
        for query_scores, query_rows in zip(scores, rows):
            hits, missing = [], []
            for rank, (score, row) in enumerate(zip(query_scores, query_rows), start=1):
                if row == -1:
                    missing.append({'rank': rank, 'status': 'faiss_missing_row'})
                    continue
                if row < 0 or row >= len(self.mapping) or not np.isfinite(score):
                    raise ValueError('FAISS returned invalid row or non-finite real score')
                hits.append({'rank': rank, 'row_id': int(row), 'document_id': self.mapping[row]['document_id'],
                             'title': self.documents[row]['title'], 'raw_score': float(score)})
            results.append({'hits': hits, 'requested_k': k, 'actual_returned_k': len(hits), 'missing_rows': missing})
        return results
