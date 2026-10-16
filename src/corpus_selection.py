"""Frozen corpus validation and deterministic selection, without native runtimes."""
import hashlib
import heapq
import json
from pathlib import Path
from src.embedding_contract import stable_json, sha256_file

def fingerprint(value):
    return hashlib.sha256(stable_json(value).encode('utf-8')).hexdigest()


def verify_corpus(corpus_dir):
    corpus_dir = Path(corpus_dir)
    manifest_path = corpus_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    checksum = fingerprint({key: value for key, value in manifest.items() if key != 'corpus_checksum'})
    if manifest.get('schema_version') != 1 or checksum != manifest['corpus_checksum']:
        raise ValueError('Corpus manifest version/checksum mismatch')
    documents = corpus_dir / 'documents.jsonl'
    if sha256_file(documents) != manifest['output_file_checksums']['documents.jsonl']:
        raise ValueError('Corpus documents checksum mismatch')
    return manifest, sha256_file(manifest_path)


def select_smoke_documents(corpus_dir, *, limit=128, seed=42):
    """Lowest SHA256(seed:document_id); O(limit) state, no gold-dependent selection."""
    if not 1 <= limit <= 1000:
        raise ValueError('Phase 3A smoke limit must be between 1 and 1,000')
    manifest, manifest_hash = verify_corpus(corpus_dir)
    heap, previous, count = [], None, 0
    with (Path(corpus_dir) / 'documents.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            identifier = row['document_id']
            if not isinstance(identifier, str) or not identifier or previous is not None and identifier <= previous:
                raise ValueError('Corpus IDs missing, duplicate or unsorted')
            if row['embedding_text'] != row['title'] + '\n' + ' '.join(row['sentences']):
                raise ValueError('Corpus embedding text rule mismatch')
            previous = identifier
            priority = int(hashlib.sha256(f'{seed}:{identifier}'.encode()).hexdigest(), 16)
            item = (-priority, identifier, row)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            count += 1
    if count != manifest['unique_passages'] or count < limit:
        raise ValueError('Corpus count mismatch or smoke limit exceeds document count')
    selected = sorted((item[2] for item in heap), key=lambda row: row['document_id'])
    selection = {'strategy': 'lowest_sha256_of_seed_colon_document_id', 'seed': seed, 'limit': limit,
                 'document_ids': [row['document_id'] for row in selected]}
    selection['selection_hash'] = fingerprint(selection)
    return selected, selection, manifest, manifest_hash


