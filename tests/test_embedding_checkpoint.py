from dataclasses import replace
import json
import numpy as np
import pytest
from src.embedding_checkpoint import (run_checkpoint, CheckpointError, CheckpointInterrupted,
                                      atomic_json, atomic_write)
from src.embedding_contract import EncoderConfig, sha256_file


class Encoder:
    def __init__(self):
        self.config = EncoderConfig()
        self.calls = []
        self.last_stats = {'synthetic': True}
    def encode_documents(self, texts):
        self.calls.extend(texts)
        values = np.zeros((len(texts), 768), np.float32)
        values[:, 0] = 1
        return values


@pytest.fixture
def job(tmp_path):
    docs = [{'document_id': f'doc-{i:03}', 'embedding_text': f'text {i}'} for i in range(7)]
    return docs, Encoder(), tmp_path/'checkpoint', {'corpus_checksum': 'corpus',
        'corpus_manifest_hash': 'manifest', 'expected_count': 7, 'shard_size': 3}


def test_interruption_and_resume_does_not_reencode_committed_rows(job):
    docs, encoder, path, kwargs = job
    with pytest.raises(CheckpointInterrupted):
        run_checkpoint(iter(docs), encoder, path, stop_after_shards=1, **kwargs)
    assert len(encoder.calls) == 3
    first = sha256_file(path/'shard-000000'/'vectors.npy')
    result = run_checkpoint(iter(docs), encoder, path, **kwargs)
    assert encoder.calls == [d['embedding_text'] for d in docs]
    assert result['row_count'] == 7 and result['resumed_shards'] == 1
    assert sha256_file(path/'shard-000000'/'vectors.npy') == first
    assert run_checkpoint(docs, encoder, path, **kwargs)['newly_encoded_shards'] == 0
    assert len(encoder.calls) == 7


def test_incomplete_shard_is_detected_and_explicit_recovery_preserves_files(job):
    docs, encoder, path, kwargs = job
    def interrupt(stage, shard):
        raise RuntimeError('power loss simulation')
    with pytest.raises(RuntimeError, match='power loss'):
        run_checkpoint(docs, encoder, path, fault_hook=interrupt, **kwargs)
    with pytest.raises(CheckpointError, match='Incomplete shard'):
        run_checkpoint(docs, encoder, path, **kwargs)
    result = run_checkpoint(docs, encoder, path, recover_incomplete=True, **kwargs)
    assert result['completion_status'] == 'complete'
    preserved = list((path/'interrupted').iterdir())
    assert len(preserved) == 1 and (preserved[0]/'vectors.npy').exists()
    assert result['recovery_events'][0]['reason'] == 'uncommitted_partial_shard'


@pytest.mark.parametrize('field', ['corpus_checksum', 'corpus_manifest_hash', 'model_revision', 'batch_size'])
def test_resume_contract_mismatch_stops_before_encoding(job, field):
    docs, encoder, path, kwargs = job
    run_checkpoint(docs, encoder, path, **kwargs)
    if field.startswith('corpus'):
        kwargs[field] = 'wrong'
    else:
        encoder.config = replace(encoder.config, **{field: 'f'*40 if field == 'model_revision' else 16})
    with pytest.raises(CheckpointError, match='config mismatch'):
        run_checkpoint(docs, encoder, path, **kwargs)
    assert len(encoder.calls) == 7


def test_completed_shard_checksum_mismatch_never_auto_repairs(job):
    docs, encoder, path, kwargs = job
    run_checkpoint(docs, encoder, path, **kwargs)
    with (path/'shard-000001'/'vectors.npy').open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(CheckpointError, match='checksum mismatch'):
        run_checkpoint(docs, encoder, path, recover_incomplete=True, **kwargs)
    assert len(encoder.calls) == 7


def test_missing_completed_shard(job):
    docs, encoder, path, kwargs = job
    run_checkpoint(docs, encoder, path, **kwargs)
    (path/'shard-000001').rename(path/'preserved_missing_shard')
    with pytest.raises(CheckpointError, match='Missing completed shard'):
        run_checkpoint(docs, encoder, path, **kwargs)


def test_duplicate_row_id_detected_even_after_checksum_update(job):
    docs, encoder, path, kwargs = job
    run_checkpoint(docs, encoder, path, **kwargs)
    folder = path/'shard-000001'
    rows = [json.loads(line) for line in (folder/'rows.jsonl').read_text().splitlines()]
    rows[1]['row_id'] = rows[0]['row_id']
    atomic_write(folder/'rows.jsonl', lambda f: f.write(('\n'.join(json.dumps(r) for r in rows)+'\n').encode()))
    meta = json.loads((folder/'metadata.json').read_text())
    meta['checksums']['rows.jsonl'] = sha256_file(folder/'rows.jsonl')
    atomic_json(folder/'metadata.json', meta)
    with pytest.raises(CheckpointError, match='Row IDs'):
        run_checkpoint(docs, encoder, path, **kwargs)


def test_document_alignment_revalidated_against_resume_input(job):
    docs, encoder, path, kwargs = job
    run_checkpoint(docs, encoder, path, **kwargs)
    changed = [dict(d) for d in docs]
    changed[-1]['document_id'] = 'doc-999'
    with pytest.raises(CheckpointError, match='alignment mismatch'):
        run_checkpoint(changed, encoder, path, **kwargs)


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'unsorted'])
def test_input_missing_duplicate_or_unsorted_ids_rejected(job, kind):
    docs, encoder, path, kwargs = job
    if kind == 'missing':
        del docs[-2:]
    elif kind == 'duplicate':
        docs[1]['document_id'] = docs[0]['document_id']
    else:
        docs.reverse()
    with pytest.raises(CheckpointError):
        run_checkpoint(docs, encoder, path, **kwargs)
    if kind == 'missing':
        assert not (path/'shard-000001'/'metadata.json').exists()


def test_atomic_failure_leaves_no_completed_file(tmp_path):
    target = tmp_path/'vectors.npy'
    def failure(handle):
        handle.write(b'partial')
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):
        atomic_write(target, failure)
    assert not target.exists()
    assert len(list(tmp_path.glob('*.tmp'))) == 1


def test_resume_after_metadata_commit_before_global_state_update(job):
    docs, encoder, path, kwargs = job
    run_checkpoint(docs, encoder, path, **kwargs)
    state = json.loads((path/'run.json').read_text())
    state.update(completion_status='in_progress', completed_shards=[])
    atomic_json(path/'run.json', state)
    result = run_checkpoint(docs, encoder, path, **kwargs)
    assert result['resumed_shards'] == 3 and result['newly_encoded_shards'] == 0
    assert len(encoder.calls) == 7


def test_full_encoding_guard_and_fixed_token_policy(job):
    docs, encoder, path, kwargs = job
    kwargs['expected_count'] = 384857
    with pytest.raises(CheckpointError, match='full encoding disabled'):
        run_checkpoint(docs, encoder, path, **kwargs)
    kwargs['expected_count'] = 7
    encoder.config = replace(encoder.config, document_max_length=8192)
    with pytest.raises(CheckpointError, match='512/512'):
        run_checkpoint(docs, encoder, path, **kwargs)
