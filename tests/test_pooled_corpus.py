"""Small offline archives exercise Phase 2 without models or FAISS."""
from copy import deepcopy
import json
from pathlib import Path
import zipfile
import pytest

from corpus.build_pooled_corpus import (HashCollisionError, canonicalize, export_prepared,
                                       passage_content, prepare)
from src.datasets.two_wiki import sha256_file


def question(identifier, title='Title', sentences=None):
    return {'_id': identifier, 'question': 'Question?', 'answer': 'GOLD_SECRET_NEVER_EXPORT',
            'type': 'compositional', 'context': [[title, sentences or ['First.', 'Second.']]],
            'supporting_facts': [[title, 0]], 'evidences': [['ANNOTATION_SECRET', 'relation', 'object']],
            'entity_ids': 'Q1_Q2'}


def source(tmp_path, train, dev=()):
    archive = tmp_path / 'official_fixture.zip'
    with zipfile.ZipFile(archive, 'x') as handle:
        handle.writestr('train.json', json.dumps(train))
        handle.writestr('dev.json', json.dumps(list(dev)))
        handle.writestr('test.json', json.dumps([question('test', 'TEST_MUST_NOT_APPEAR')]))
    return archive, {'sha256': sha256_file(archive), 'source_url': 'https://example.invalid/fixture',
                     'version': 'fixture-v1', 'original_filename': archive.name}


def build(archive, provenance, root, *, policy='train_plus_validation'):
    dry = prepare(archive, provenance, root / 'work', policy=policy)
    result = export_prepared(dry['prepared_database'], root / 'assets',
                             expected_database_sha256=dry['prepared_database_sha256'])
    return dry, result, Path(result['output_directory'])


def records(directory, filename):
    return [json.loads(line) for line in (directory / filename).read_text(encoding='utf-8').splitlines()]


def test_full_content_dedup_same_title_and_cross_split_overlap(tmp_path):
    archive, provenance = source(tmp_path, [question('a'), question('b', sentences=['Different.'])], [question('c')])
    dry, result, directory = build(archive, provenance, tmp_path / 'build')
    manifest = result['manifest']
    assert manifest['total_context_entries'] == 3
    assert manifest['unique_passages'] == 2 and manifest['duplicate_count'] == 1
    assert manifest['cross_split_overlap']['shared_unique_passages'] == 1
    assert manifest['cross_split_overlap']['source_entries_on_shared_passages'] == 2
    assert len(records(directory, 'documents.jsonl')) == 2
    refs = records(directory, 'source_mappings.jsonl')
    assert len(refs) == 3 and {r['source_question_id'] for r in refs} == {'a', 'b', 'c'}
    assert {r['source_split'] for r in refs} == {'train', 'dev'}
    assert all(r['context_position'] == 0 for r in refs)
    assert sum(dry['estimated_output_bytes_by_file'].values()) == sum(manifest['output_file_sizes_bytes'].values())


def test_canonical_whitespace_nfc_and_raw_sentence_variants_preserved(tmp_path):
    first = question('a', 'Cafe\u0301', [' First.', 'Second.'])
    second = question('b', 'Café', ['First.   Second.'])
    archive, provenance = source(tmp_path, [first, second])
    _, result, directory = build(archive, provenance, tmp_path / 'build')
    assert result['manifest']['unique_passages'] == 1
    assert result['manifest']['unique_raw_variants'] == 2
    variants = {v['variant_id']: v for v in records(directory, 'source_variants.jsonl')}
    refs = records(directory, 'source_mappings.jsonl')
    for ref in refs:
        original = first if ref['source_question_id'] == 'a' else second
        variant = variants[ref['variant_id']]
        assert [variant['title'], variant['sentences']] == original['context'][0]
        assert variant['embedding_text'] == original['context'][0][0] + '\n' + ' '.join(original['context'][0][1])


def test_same_text_different_titles_not_deduplicated(tmp_path):
    archive, provenance = source(tmp_path, [question('a', 'One'), question('b', 'Two')])
    _, result, _ = build(archive, provenance, tmp_path / 'build')
    assert result['manifest']['unique_passages'] == 2


def test_rebuild_identical_order_ids_and_file_checksums(tmp_path):
    archive, provenance = source(tmp_path, [question('z', 'ZZZ'), question('a', 'AAA')], [question('d', 'AAA')])
    _, first, path1 = build(archive, provenance, tmp_path / 'first')
    _, second, path2 = build(archive, provenance, tmp_path / 'second')
    assert first['corpus_checksum'] == second['corpus_checksum']
    for name in [*first['manifest']['output_file_checksums'], 'manifest.json']:
        assert (path1 / name).read_bytes() == (path2 / name).read_bytes()
    ids = [d['document_id'] for d in records(path1, 'documents.jsonl')]
    assert ids == sorted(ids)
    refs = records(path1, 'source_mappings.jsonl')
    assert refs == sorted(refs, key=lambda r: (r['document_id'], r['source_split'], r['source_question_id'], r['context_position']))


def test_train_only_preserves_shared_document_ids_and_excludes_dev_test(tmp_path):
    archive, provenance = source(tmp_path, [question('a')], [question('b'), question('c', 'DEV_ONLY')])
    _, both, both_dir = build(archive, provenance, tmp_path / 'both')
    _, train, train_dir = build(archive, provenance, tmp_path / 'train', policy='train_only')
    train_docs = records(train_dir, 'documents.jsonl')
    assert len(train_docs) == 1
    assert train_docs[0]['document_id'] in {d['document_id'] for d in records(both_dir, 'documents.jsonl')}
    assert train['manifest']['included_splits'] == ['train']
    assert train['manifest']['cross_split_overlap']['shared_unique_passages'] == 0
    assert both['manifest']['included_splits'] == ['train', 'dev']
    assert 'TEST_MUST_NOT_APPEAR' not in (both_dir / 'documents.jsonl').read_text(encoding='utf-8')


def test_evidence_invalid_and_qa_invalid_do_not_filter_context(tmp_path):
    first = question('a')
    first['supporting_facts'] = [['Title', 99]]
    second = question('b', 'Second title')
    second['answer'] = ''
    archive, provenance = source(tmp_path, [first, second])
    _, result, directory = build(archive, provenance, tmp_path / 'build')
    assert result['manifest']['total_context_entries'] == 2
    refs = {r['source_question_id']: r for r in records(directory, 'source_mappings.jsonl')}
    assert refs['a']['qa_valid'] and not refs['a']['evidence_valid']
    assert refs['b']['qa_valid'] is False
    assert result['manifest']['split_statistics']['train']['evidence_invalid_context_entries_included'] == 1


def test_forced_hash_collision_aborts_without_merging(tmp_path):
    archive, provenance = source(tmp_path, [question('a', 'One'), question('b', 'Two')])
    with pytest.raises(HashCollisionError, match='collision'):
        prepare(archive, provenance, tmp_path / 'work', hash_fn=lambda _: 'forced-collision')


def test_missing_input_and_hash_mismatch(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare(tmp_path / 'missing.zip', {}, tmp_path / 'missing_work')
    archive, provenance = source(tmp_path, [question('a')])
    provenance['sha256'] = 'wrong'
    with pytest.raises(ValueError, match='SHA256'):
        prepare(archive, provenance, tmp_path / 'bad_work')


def test_duplicate_question_ids_and_malformed_context_stop(tmp_path):
    archive, provenance = source(tmp_path, [question('a'), question('a')])
    with pytest.raises(ValueError, match='Duplicate question ID'):
        prepare(archive, provenance, tmp_path / 'work')


def test_malformed_empty_sentence_list_not_silently_dropped(tmp_path):
    row = question('a')
    row['context'][0][1] = []
    archive, provenance = source(tmp_path, [row])
    with pytest.raises(ValueError, match='Malformed context'):
        prepare(archive, provenance, tmp_path / 'work')


def test_no_gold_leakage_and_raw_archive_unchanged(tmp_path):
    archive, provenance = source(tmp_path, [question('a')])
    digest = sha256_file(archive)
    _, _, directory = build(archive, provenance, tmp_path / 'build')
    for path in directory.iterdir():
        text = path.read_text(encoding='utf-8')
        assert 'GOLD_SECRET_NEVER_EXPORT' not in text
        assert 'ANNOTATION_SECRET' not in text
    assert sha256_file(archive) == digest


def test_mapping_traces_every_entry_and_context_position(tmp_path):
    row = question('a')
    row['context'].append(deepcopy(row['context'][0]))
    archive, provenance = source(tmp_path, [row])
    _, result, directory = build(archive, provenance, tmp_path / 'build')
    refs = records(directory, 'source_mappings.jsonl')
    assert [r['context_position'] for r in refs] == [0, 1]
    assert refs[0]['document_id'] == refs[1]['document_id']
    assert refs[0]['source_row_index'] == 0
    metadata = json.loads((directory / 'provenance.json').read_text(encoding='utf-8'))
    assert metadata[refs[0]['source_provenance_id']]['dataset_sha256'] == provenance['sha256']


def test_prepared_state_tampering_and_existing_outputs_not_overwritten(tmp_path):
    archive, provenance = source(tmp_path, [question('a')])
    dry, result, directory = build(archive, provenance, tmp_path / 'build')
    repeat = export_prepared(dry['prepared_database'], tmp_path / 'build' / 'assets',
                             expected_database_sha256=dry['prepared_database_sha256'])
    assert repeat['newly_published'] is False
    assert repeat['corpus_checksum'] == result['corpus_checksum']
    assert directory.exists()
    with pytest.raises(ValueError, match='database differs'):
        export_prepared(dry['prepared_database'], tmp_path / 'unused', expected_database_sha256='wrong')


def test_hotpotqa_regression_after_corpus_import(tmp_path):
    from src.datasets import load_dataset
    row = {'_id': 'hp', 'question': 'Original?', 'answer': 'answer', 'context': []}
    path = tmp_path / 'hotpot.json'
    path.write_text(json.dumps([row]), encoding='utf-8')
    assert load_dataset('hotpotqa', path) == [row]


def test_canonical_payload_has_unambiguous_title_text_boundaries():
    assert passage_content('a b', ['c'])[0] != passage_content('a', ['b c'])[0]
    assert canonicalize(' Cafe\u0301\t X\n') == 'Café X'


def test_verifier_reconstructs_every_raw_context_and_rejects_tampering(tmp_path):
    from scripts.verify_pooled_corpus import verify
    archive, provenance = source(tmp_path, [question('a')], [question('b', 'Another')])
    dry, result, directory = build(archive, provenance, tmp_path / 'build')
    report = verify(directory, dry['prepared_database'], archive)
    assert report['every_original_context_entry_preserved']
    assert report['raw_context_entries_reconstructed_by_split'] == {'train': 1, 'dev': 1}
    with (directory / 'documents.jsonl').open('ab') as handle:
        handle.write(b'{}\n')
    with pytest.raises(ValueError, match='Artifact checksum'):
        verify(directory, dry['prepared_database'], archive)
