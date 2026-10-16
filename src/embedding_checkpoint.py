"""Bounded Phase 3B-0 shard writer; no model, FAISS or full-corpus entrypoint."""
from dataclasses import asdict
import json
from importlib.metadata import version
import math
import os
from pathlib import Path
import sys
import uuid
import numpy as np
from src.embedding_contract import sha256_file, stable_json, validate_vectors
from src.corpus_selection import fingerprint


class CheckpointError(ValueError):
    pass


class CheckpointInterrupted(RuntimeError):
    pass


def local_path(value):
    path = Path(value).resolve()
    return Path('\\\\?\\' + str(path)) if sys.platform == 'win32' and not str(path).startswith('\\\\?\\') else path


def atomic_write(path, writer):
    """Commit within one directory/filesystem; leave failed temps for audit."""
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as handle:
        writer(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_write(path, lambda handle: handle.write((stable_json(value)+'\n').encode('utf-8')))


def _validate_shard(directory, contract, shard_id):
    try:
        metadata = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        start = shard_id * contract['shard_size']
        count = min(contract['shard_size'], contract['expected_row_count'] - start)
        expected = {'schema_version': 1, 'completion_status': 'complete', 'shard_id': shard_id,
                    'row_start': start, 'row_count': count, 'shape': [count, contract['embedding_dimension']],
                    **{key: contract[key] for key in ('corpus_checksum', 'corpus_manifest_hash', 'model_id',
                       'model_revision', 'encoder_configuration_hash', 'embedding_dimension', 'embedding_dtype')}}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise CheckpointError(f'Shard metadata/config mismatch: {shard_id}')
        for name in ('vectors.npy', 'rows.jsonl'):
            if sha256_file(directory/name) != metadata['checksums'][name]:
                raise CheckpointError(f'Shard checksum mismatch: {shard_id}/{name}')
        rows = [json.loads(line) for line in (directory/'rows.jsonl').read_text(encoding='utf-8').splitlines()]
        ids = [row['document_id'] for row in rows]
        if [row['row_id'] for row in rows] != list(range(start, start+count)):
            raise CheckpointError(f'Duplicate/missing/out-of-order Row IDs in shard {shard_id}')
        if len(ids) != count or any(not isinstance(i, str) or not i for i in ids) or ids != sorted(set(ids)):
            raise CheckpointError(f'Duplicate/missing/out-of-order Document IDs in shard {shard_id}')
        if metadata['document_id_range'] != [ids[0], ids[-1]] or metadata['document_ids_sha256'] != fingerprint(ids):
            raise CheckpointError(f'Document ID range/hash mismatch in shard {shard_id}')
        vectors = np.load(directory/'vectors.npy', allow_pickle=False, mmap_mode='r')
        if list(vectors.shape) != expected['shape']:
            raise CheckpointError(f'Shard shape mismatch: {shard_id}')
        validate_vectors(vectors, contract['embedding_dimension'])
        return metadata, ids
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f'Incomplete/malformed shard {shard_id}: {exc}') from exc


def run_checkpoint(documents, encoder, output_dir, *, corpus_checksum, corpus_manifest_hash,
                   expected_count, shard_size=256, selection_hash=None,
                   recover_incomplete=False, stop_after_shards=None, fault_hook=None):
    """Resume validates every committed shard before any encoding; never skips rows.

    Explicit recovery quarantines an uncommitted shard, without deleting files.
    Completed-but-corrupt shards always fail and are never silently re-encoded.
    The 5,000 guard remains until a separate Full Embedding authorization.
    """
    if type(expected_count) is not int or not 1 <= expected_count <= 5000:
        raise CheckpointError('Phase 3B-0 permits at most 5,000 documents; full encoding disabled')
    if type(shard_size) is not int or shard_size < 1:
        raise CheckpointError('Invalid shard size')
    config = asdict(encoder.config)
    if config['document_max_length'] != 512 or config['query_max_length'] != 512:
        raise CheckpointError('Phase 3B-0 token policy is fixed at 512/512')
    contract = {'schema_version': 1, 'scope': 'benchmark', 'full_embedding_authorized': False,
        'corpus_checksum': corpus_checksum, 'corpus_manifest_hash': corpus_manifest_hash,
        'model_id': config['model_id'], 'model_revision': config['model_revision'],
        'encoder_configuration': config, 'encoder_configuration_hash': encoder.config.config_hash,
        'embedding_dimension': config['dimension'], 'embedding_dtype': 'float32',
        'expected_row_count': expected_count, 'shard_size': shard_size, 'selection_hash': selection_hash,
        'runtime_versions': {name: version(name) for name in ('torch', 'transformers', 'numpy')},
        'ordering': 'strictly ascending original corpus document_id'}
    root = local_path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_file = root/'run.json'
    if run_file.exists():
        state = json.loads(run_file.read_text(encoding='utf-8'))
        if state.get('contract') != contract:
            raise CheckpointError('Resume corpus/model/encoder/selection/config mismatch')
        if state.get('completion_status') not in {'in_progress', 'complete'}:
            raise CheckpointError('Invalid checkpoint completion status')
    else:
        if any(root.iterdir()):
            raise CheckpointError('Uninitialized nonempty checkpoint directory')
        state = {'contract': contract, 'completion_status': 'in_progress', 'completed_shards': [],
                 'recovery_events': []}
        atomic_json(run_file, state)
    expected_shards = math.ceil(expected_count/shard_size)
    committed = {}
    declared = set(state['completed_shards'])
    if len(declared) != len(state['completed_shards']) or any(type(i) is not int or not 0 <= i < expected_shards for i in declared):
        raise CheckpointError('Invalid/duplicate completed shard IDs')
    for shard_id in declared:
        if not (root/f'shard-{shard_id:06d}').is_dir():
            raise CheckpointError(f'Missing completed shard: {shard_id}')
    for directory in sorted(root.glob('shard-*')):
        try:
            shard_id = int(directory.name.removeprefix('shard-'))
        except ValueError as exc:
            raise CheckpointError('Unexpected shard name') from exc
        if directory.name != f'shard-{shard_id:06d}' or not 0 <= shard_id < expected_shards:
            raise CheckpointError('Unexpected shard ID')
        if not (directory/'metadata.json').exists():
            if not recover_incomplete or shard_id in declared or state['completion_status'] == 'complete':
                raise CheckpointError(f'Incomplete shard: {shard_id}; explicit recovery required')
            quarantine = root/'interrupted'
            quarantine.mkdir(exist_ok=True)
            target = quarantine/(directory.name+'-'+uuid.uuid4().hex)
            os.replace(directory, target)
            state['recovery_events'].append({'shard_id': shard_id, 'reason': 'uncommitted_partial_shard',
                                             'preserved_at': str(target)})
        else:
            committed[shard_id] = _validate_shard(directory, contract, shard_id)
    if state['completion_status'] == 'complete' and set(committed) != set(range(expected_shards)):
        raise CheckpointError('Missing shard from completed checkpoint')
    state['completed_shards'] = sorted(committed)
    atomic_json(run_file, state)
    buffer, previous, count, newly_encoded = [], None, 0, 0
    all_ids = []
    def consume(rows, shard_id):
        nonlocal newly_encoded
        ids = [row['document_id'] for row in rows]
        if shard_id in committed:
            if committed[shard_id][1] != ids:
                raise CheckpointError('Final Embedding Row ID / Corpus Document ID alignment mismatch')
            return
        vectors = encoder.encode_documents([row['embedding_text'] for row in rows])
        validate_vectors(vectors, config['dimension'])
        if vectors.shape != (len(rows), config['dimension']):
            raise CheckpointError('Encoder row count mismatch')
        directory = root/f'shard-{shard_id:06d}'
        directory.mkdir(exist_ok=False)
        atomic_write(directory/'vectors.npy', lambda handle: np.save(handle, vectors, allow_pickle=False))
        mapping = [{'row_id': shard_id*shard_size+i, 'document_id': identifier} for i, identifier in enumerate(ids)]
        atomic_write(directory/'rows.jsonl', lambda handle: handle.write(('\n'.join(stable_json(r) for r in mapping)+'\n').encode('utf-8')))
        if fault_hook:
            fault_hook('before_metadata_commit', shard_id)
        metadata = {'schema_version': 1, 'completion_status': 'complete', 'shard_id': shard_id,
            **{key: contract[key] for key in ('corpus_checksum', 'corpus_manifest_hash', 'model_id', 'model_revision',
               'encoder_configuration_hash', 'embedding_dimension', 'embedding_dtype')},
            'row_start': shard_id*shard_size, 'row_count': len(ids), 'shape': list(vectors.shape),
            'document_id_range': [ids[0], ids[-1]], 'document_ids_sha256': fingerprint(ids),
            'checksums': {name: sha256_file(directory/name) for name in ('vectors.npy', 'rows.jsonl')},
            'encoding_statistics': encoder.last_stats}
        # Metadata is the commit marker, written after both data files are durable.
        atomic_json(directory/'metadata.json', metadata)
        committed[shard_id] = (metadata, ids)
        state['completed_shards'] = sorted(committed)
        atomic_json(run_file, state)
        newly_encoded += 1
        if stop_after_shards is not None and newly_encoded >= stop_after_shards:
            raise CheckpointInterrupted('Simulated interruption after committed shard')
    for row in documents:
        identifier = row.get('document_id')
        if not isinstance(identifier, str) or not identifier or previous is not None and identifier <= previous:
            raise CheckpointError('Corpus Document IDs duplicate/missing/out of order')
        if count >= expected_count:
            raise CheckpointError('Input exceeds declared row count')
        if not isinstance(row.get('embedding_text'), str) or not row['embedding_text'].strip():
            raise CheckpointError('Missing embedding text')
        previous = identifier
        all_ids.append(identifier)
        buffer.append(row)
        count += 1
        if len(buffer) == shard_size:
            consume(buffer, (count-1)//shard_size)
            buffer = []
    if count != expected_count:
        raise CheckpointError('Missing documents at final validation; partial final shard not committed')
    if buffer:
        consume(buffer, (count-1)//shard_size)
    if set(committed) != set(range(expected_shards)):
        raise CheckpointError('Missing documents/shards at final validation')
    # Re-read all persisted rows/vectors and compare the exact corpus order.
    persisted_ids = []
    for shard_id in range(expected_shards):
        metadata, ids = _validate_shard(root/f'shard-{shard_id:06d}', contract, shard_id)
        persisted_ids.extend(ids)
    if persisted_ids != all_ids:
        raise CheckpointError('Final Embedding Row ID / Corpus Document ID alignment mismatch')
    state.update(completion_status='complete', row_count=count, document_ids_sha256=fingerprint(all_ids),
                 completed_shards=list(range(expected_shards)))
    atomic_json(run_file, state)
    return {'completion_status': 'complete', 'row_count': count, 'shard_count': expected_shards,
            'newly_encoded_shards': newly_encoded, 'resumed_shards': expected_shards-newly_encoded,
            'document_ids_sha256': fingerprint(all_ids), 'recovery_events': state['recovery_events']}
