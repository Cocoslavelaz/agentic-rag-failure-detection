from dataclasses import asdict, replace
import hashlib
import json
import numpy as np
import pytest
from src.embedding_contract import EncoderConfig, sha256_file, stable_json
from src.faiss_process import ProcessFaissIndex, build_index
from src.corpus_selection import fingerprint, select_smoke_documents
from src.local_retriever import LocalRetriever, unexecuted_retrieval


@pytest.fixture
def assets(tmp_path):
    vectors = np.eye(768, dtype=np.float32)[:3]
    docs = [{'document_id': f'doc-{i}', 'title': f'Title {i}', 'sentences': ['text'],
             'embedding_text': f'Title {i}\ntext'} for i in range(3)]
    config = EncoderConfig()
    directory = tmp_path / 'index'
    selection = {'document_ids': [d['document_id'] for d in docs]}
    manifest, seconds = build_index(directory, docs, vectors, corpus_checksum='corpus',
        corpus_manifest_hash='manifest', encoder_config=config, selection=selection)
    kwargs = dict(expected_corpus_checksum='corpus', expected_corpus_manifest_hash='manifest',
                  expected_encoder_config=config)
    return directory, docs, vectors, config, kwargs


def rehash(directory, mutate):
    path = directory / 'index_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    mutate(manifest)
    path.write_text(stable_json(manifest), encoding='utf-8')
    (directory / 'build_complete.json').write_text(stable_json({'status': 'complete',
        'index_manifest_sha256': sha256_file(path)}), encoding='utf-8')


def test_round_trip_alignment_scores_cosine_and_missing_rows(assets):
    directory, docs, vectors, config, kwargs = assets
    index = ProcessFaissIndex.load(directory, **kwargs)
    query = np.zeros((1, 768), np.float32)
    query[0, :3] = [0.8, 0.6, 0]
    result = index.search(query, k=5)[0]
    assert [hit['document_id'] for hit in result['hits'][:2]] == ['doc-0', 'doc-1']
    assert [hit['raw_score'] for hit in result['hits'][:2]] == [float(np.float32(.8)), float(np.float32(.6))]
    assert result['actual_returned_k'] == 3
    assert result['missing_rows'] == [{'rank': 4, 'status': 'faiss_missing_row'},
                                      {'rank': 5, 'status': 'faiss_missing_row'}]
    check = index.compare_cosine(query, k=5)
    assert check['verified'] and check['max_absolute_error'] < 1e-5
    assert check['topk_exact_row_order_match'] == [True]
    assert index.documents == docs and index.manifest['scope'] == 'smoke'
    assert index.manifest['research_retrieval_index'] is False


@pytest.mark.parametrize('key,value', [('expected_corpus_checksum', 'wrong'),
                                     ('expected_corpus_manifest_hash', 'wrong'),
                                     ('expected_encoder_config', replace(EncoderConfig(), model_revision='f'*40))])
def test_index_rejects_version_mismatch(assets, key, value):
    directory, _, _, _, kwargs = assets
    kwargs[key] = value
    with pytest.raises(ValueError, match='mismatch'):
        ProcessFaissIndex.load(directory, **kwargs)


def test_ntotal_mismatch(assets):
    directory, _, _, _, kwargs = assets
    rehash(directory, lambda m: m.update(number_of_documents=4))
    with pytest.raises(ValueError, match='ntotal'):
        ProcessFaissIndex.load(directory, **kwargs)


def test_mapping_alignment_and_checksum(assets):
    directory, _, _, _, kwargs = assets
    path = directory / 'document_mapping.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    rows[0]['row_id'] = 2
    path.write_text('\n'.join(stable_json(row) for row in rows), encoding='utf-8')
    with pytest.raises(ValueError, match='checksum'):
        ProcessFaissIndex.load(directory, **kwargs)
    rehash(directory, lambda m: m['file_checksums'].update({'document_mapping.jsonl': sha256_file(path)}))
    with pytest.raises(ValueError, match='alignment'):
        ProcessFaissIndex.load(directory, **kwargs)


@pytest.mark.parametrize('identifier', [None, '', 'doc-1'])
def test_duplicate_or_missing_document_ids(tmp_path, identifier):
    docs = [{'document_id': identifier}, {'document_id': 'doc-1'}]
    with pytest.raises(ValueError, match='Duplicate/missing'):
        build_index(tmp_path / 'bad', docs, np.eye(768, dtype=np.float32)[:2],
            corpus_checksum='c', corpus_manifest_hash='m', encoder_config=EncoderConfig(),
            selection={'document_ids': [identifier, 'doc-1']})
    assert not (tmp_path / 'bad').exists()


def test_deterministic_selection_without_labels(tmp_path):
    docs = [{'document_id': str(i), 'title': str(i), 'sentences': ['text'],
             'embedding_text': str(i)+'\ntext'} for i in range(9)]
    path = tmp_path / 'documents.jsonl'
    path.write_text('\n'.join(stable_json(doc) for doc in docs), encoding='utf-8')
    manifest = {'schema_version': 1, 'unique_passages': 9,
                'output_file_checksums': {'documents.jsonl': sha256_file(path)}}
    manifest['corpus_checksum'] = fingerprint(manifest)
    (tmp_path / 'manifest.json').write_text(stable_json(manifest), encoding='utf-8')
    a = select_smoke_documents(tmp_path, limit=3, seed=42)
    b = select_smoke_documents(tmp_path, limit=3, seed=42)
    assert a == b
    expected = sorted(docs, key=lambda d: hashlib.sha256(('42:'+d['document_id']).encode()).hexdigest())[:3]
    assert a[1]['document_ids'] == sorted(d['document_id'] for d in expected)


def test_windows_long_unicode_path_round_trip(assets, tmp_path):
    _, docs, vectors, config, kwargs = assets
    directory = tmp_path / ('a' * 90) / ('b' * 90) / '\u7d22\u5f15'
    build_index(directory, docs, vectors, corpus_checksum='corpus',
        corpus_manifest_hash='manifest', encoder_config=config,
        selection={'document_ids': [doc['document_id'] for doc in docs]})
    index = ProcessFaissIndex.load(directory, **kwargs)
    assert index.search(vectors[:1], k=1)[0]['hits'][0]['document_id'] == 'doc-0'


def test_retriever_telemetry_uses_only_query_and_raw_scores(assets):
    directory, _, _, config, kwargs = assets
    class Encoder:
        def __init__(self):
            self.config = config
            self.inputs = []
        def encode_queries(self, texts):
            self.inputs.extend(texts)
            q = np.zeros((1, 768), np.float32)
            q[0, :2] = [.8, .6]
            return q
    encoder = Encoder()
    retriever = LocalRetriever(encoder, ProcessFaissIndex.load(directory, **kwargs), top_k=2)
    event = retriever.search('plain question')
    assert encoder.inputs == ['plain question']
    assert event['raw_faiss_scores'] == event['cosine_similarities']
    assert event['top1_uncertainty'] == 1 - event['raw_faiss_scores'][0]
    assert event['topk_mean_uncertainty'] == 1 - np.mean(event['raw_faiss_scores'])
    assert event['score_margin'] == event['raw_faiss_scores'][0] - event['raw_faiss_scores'][1]
    absent = unexecuted_retrieval()
    assert absent['retrieval_occurred'] is False
    assert absent['raw_faiss_scores'] is None and absent['score_margin'] is None
