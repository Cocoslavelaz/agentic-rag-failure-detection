"""Offline fixtures reflect the inspected official release; never download data."""
from copy import deepcopy
import json
import zipfile
import pytest
from src.datasets import agent_input, load_dataset
from src.datasets.two_wiki import (DatasetValidationError, normalize_example,
                                   iter_two_wiki, load_two_wiki, sha256_file)


@pytest.fixture
def row():
    return {'_id': 'original-id', 'question': 'Who is A?', 'answer': 'B',
            'type': 'compositional', 'context': [['Title A', ['First.', 'Second.']], ['Title B', ['Third.']]],
            'supporting_facts': [['Title A', 1], ['Title B', 0]],
            'evidences': [['A', 'relation', 'B']], 'entity_ids': 'Q1_Q2',
            'answer_id': None, 'evidences_id': [], 'extra_field': {'keep': True}}


def write_rows(tmp_path, rows):
    path = tmp_path / 'arbitrary_name.json'
    path.write_text(json.dumps(rows), encoding='utf-8')
    return path


@pytest.mark.parametrize('kind', ['comparison', 'inference', 'compositional', 'bridge_comparison'])
def test_parse_preserves_fields_and_order(row, kind):
    row['type'] = kind
    original = deepcopy(row)
    result = normalize_example(row, split='validation', provenance={'version': 'fixture'}, row_index=3)
    assert result['question_id'] == 'original-id'
    assert result['question_type'] == kind and result['source_split'] == 'dev'
    assert result['context_entries'][0] == {'title': 'Title A', 'sentences': ['First.', 'Second.']}
    assert result['supporting_facts'] == [['Title A', 1], ['Title B', 0]]
    assert result['evidence_annotations'] == [['A', 'relation', 'B']]
    assert result['raw_example'] == original and row == original
    assert result['correctness_evaluation_available'] is True
    assert result['source_provenance']['row_index'] == 3
    result['raw_example']['extra_field']['keep'] = False
    result['context_entries'][0]['sentences'][0] = 'Changed'
    assert row == original


@pytest.mark.parametrize('key', ['_id', 'question', 'answer', 'type', 'context', 'supporting_facts', 'evidences', 'entity_ids'])
def test_missing_fields_error(row, key):
    del row[key]
    with pytest.raises(DatasetValidationError, match='missing_field'):
        normalize_example(row, split='train')


@pytest.mark.parametrize('answer', ['', None, 1])
def test_invalid_labeled_answer(row, answer):
    row['answer'] = answer
    with pytest.raises(DatasetValidationError, match='invalid_answer'):
        normalize_example(row, split='dev')


def test_empty_context(row):
    row['context'] = []
    with pytest.raises(DatasetValidationError, match='invalid_context'):
        normalize_example(row, split='dev')


@pytest.mark.parametrize('facts', [None, ['Title A'], [['Title A']], [['Title A', '1']], [['Title A', True]]])
def test_malformed_supporting_facts(row, facts):
    row['supporting_facts'] = facts
    with pytest.raises(DatasetValidationError, match='malformed_supporting_fact'):
        normalize_example(row, split='dev')


@pytest.mark.parametrize('index', [-1, 2, 19])
def test_invalid_sentence_index_is_not_repaired(row, index):
    row['supporting_facts'] = [['Title A', index]]
    before = deepcopy(row)
    with pytest.raises(DatasetValidationError, match='invalid_sentence_index'):
        normalize_example(row, split='dev')
    assert row == before


def test_missing_and_ambiguous_title(row):
    row['supporting_facts'] = [['Unknown', 0]]
    with pytest.raises(DatasetValidationError, match='support_title_missing'):
        normalize_example(row, split='dev')
    row['supporting_facts'] = [['Title A', 0]]
    row['context'].append(['Title A', ['Different content']])
    with pytest.raises(DatasetValidationError, match='support_title_ambiguous'):
        normalize_example(row, split='dev')


def test_duplicate_question_ids(row, tmp_path):
    path = write_rows(tmp_path, [row, row])
    with pytest.raises(DatasetValidationError, match='duplicate_question_id'):
        load_two_wiki(path, split='dev')


@pytest.mark.parametrize('placeholders', [True, False])
def test_official_unlabeled_test(row, placeholders):
    for key, value in {'answer': '', 'answer_id': '', 'entity_ids': '', 'supporting_facts': [],
                       'evidences': [], 'evidences_id': []}.items():
        if placeholders:
            row[key] = value
        else:
            row.pop(key)
    result = normalize_example(row, split='test')
    assert result['gold_answer'] is None
    assert result['correctness_evaluation_available'] is False
    assert result['answerability'] == 'unknown'
    assert result['raw_example'] == row


def test_cannot_mislabel_labeled_data_as_official_test(row):
    with pytest.raises(DatasetValidationError, match='unexpected_test_annotation'):
        normalize_example(row, split='test')


def test_gold_leakage_prevention_and_no_evidence_as_documents(row):
    result = normalize_example(row, split='dev')
    assert agent_input(result, dataset_format='2wikimultihopqa') == {'original_question': 'Who is A?'}
    assert len(result['context_entries']) == 2  # evidence triple is annotation only
    assert all('relation' not in entry['sentences'] for entry in result['context_entries'])


def test_registry_and_provenance(row, tmp_path):
    path = write_rows(tmp_path, [row])
    digest = sha256_file(path)
    result = load_dataset('2wikimultihopqa', path, split='dev', provenance={'version': 'fixture'})[0]
    assert result['source_provenance']['source_sha256'] == digest
    assert result['source_provenance']['original_filename'] == path.name
    assert result['source_provenance']['version'] == 'fixture'
    assert sha256_file(path) == digest
    with pytest.raises(ValueError, match='Explicit split'):
        load_dataset('2wikimultihopqa', path)
    with pytest.raises(ValueError, match='Unknown dataset'):
        load_dataset('guess_from_filename', path)


def test_zip_source_selection_and_missing_file(row, tmp_path):
    path = tmp_path / 'source.zip'
    with zipfile.ZipFile(path, 'x') as archive:
        archive.writestr('nested/dev.json', json.dumps([row]))
    result = load_two_wiki(path, split='validation')[0]
    assert result['source_provenance']['archive_member'] == 'nested/dev.json'
    assert len(result['source_provenance']['member_sha256']) == 64
    with pytest.raises(ValueError, match='exactly one'):
        load_two_wiki(path, split='train')
    with pytest.raises(FileNotFoundError):
        load_two_wiki(tmp_path / 'missing.json', split='dev')


def test_no_silent_skip(row, tmp_path):
    bad = deepcopy(row)
    bad['_id'] = 'bad'
    bad['supporting_facts'] = [['Title A', 99]]
    iterator = iter_two_wiki(write_rows(tmp_path, [row, bad]), split='dev')
    assert next(iterator)['question_id'] == 'original-id'
    with pytest.raises(DatasetValidationError) as error:
        next(iterator)
    assert error.value.row_index == 1 and error.value.question_id == 'bad'


@pytest.mark.parametrize('backend,format_,record', [
    ('hotpotqa', 'hotpotqa', {'_id': 'hp', 'question': 'Original?', 'answer': 'secret'}),
    ('hotpotqa', 'kilt', {'id': 'kilt', 'input': 'Original?', 'output': [{'answer': 'secret'}]}),
    ('kilt', 'kilt', {'id': 'kilt', 'input': 'Original?'})])
def test_hotpotqa_kilt_regression(tmp_path, backend, format_, record):
    path = write_rows(tmp_path, [record])
    from src.utils import load_dataset as existing_loader
    assert existing_loader(backend, path) == [record]
    assert agent_input(record, dataset_format=format_) == {'original_question': 'Original?'}


def test_validator_reports_all_errors_and_cross_split_ids(row):
    from scripts.validate_two_wiki import validate_split
    broken = deepcopy(row)
    broken['_id'] = 'broken'
    broken['supporting_facts'] = [['Title A', 5], ['Title B', -1]]
    stats = validate_split([row, broken, row], 'dev', {}, {'original-id'})
    assert stats['sample_count'] == 3
    assert stats['issue_counts']['invalid_sentence_index'] == 2
    assert stats['duplicate_id_occurrences'] == 1
    assert stats['cross_split_duplicate_occurrences'] == 2
    assert stats['supporting_fact_mapping']['unmatched'] == 2


def test_qa_valid_evidence_invalid_policy_preserves_all_annotations(row, tmp_path):
    from src.datasets.two_wiki import evaluation_eligibility
    row['supporting_facts'].append(['Title A', 99])
    original = deepcopy(row)
    result = load_dataset('2wikimultihopqa', write_rows(tmp_path, [row]), split='dev',
                          evidence_policy='retain')[0]
    assert result['qa_valid'] is True and result['evidence_valid'] is False
    assert result['raw_example'] == original
    assert result['supporting_facts'] == original['supporting_facts']
    assert result['gold_answer'] == 'B'
    assert agent_input(result, dataset_format='2wikimultihopqa') == {'original_question': 'Who is A?'}
    policy = evaluation_eligibility(result)
    assert policy['qa_metrics_eligible'] and policy['failure_prediction_eligible']
    assert not policy['annotation_dependent_evidence_metrics_eligible']
    assert any('invalid_sentence_index' in reason for reason in policy['evidence_exclusion_reasons'])
    assert policy['qa_exclusion_reasons'] == []


@pytest.mark.parametrize('change', ['missing_title', 'empty_sentences', 'out_of_range', 'malformed_fact'])
def test_raw_independent_cause_classification(row, change):
    from scripts.investigate_two_wiki_quality import raw_facts
    row['supporting_facts'] = [['Title A', 0]]
    expected = {'missing_title': 'missing_context_title', 'empty_sentences': 'empty_sentence_list',
                'out_of_range': 'sentence_index_out_of_range', 'malformed_fact': 'other'}[change]
    if change == 'missing_title':
        row['supporting_facts'][0][0] = 'Missing'
    elif change == 'empty_sentences':
        row['context'][0][1] = []
    elif change == 'out_of_range':
        row['supporting_facts'][0][1] = 20
    else:
        row['supporting_facts'] = [['Title A']]
    assert raw_facts(row)[0]['cause'] == expected


@pytest.mark.parametrize('key', ['answer', 'question', 'context'])
def test_retention_does_not_accept_broken_qa(row, key):
    from src.datasets.two_wiki import validation_flags
    row[key] = '' if key != 'context' else []
    assert validation_flags(row, split='dev')['qa_valid'] is False
    with pytest.raises(DatasetValidationError):
        normalize_example(row, split='dev', evidence_policy='retain')


def test_missing_annotation_retention_is_explicit(row):
    del row['supporting_facts']
    with pytest.raises(DatasetValidationError):
        normalize_example(row, split='dev')
    example = normalize_example(row, split='dev', evidence_policy='retain')
    assert example['qa_valid'] and not example['evidence_valid']
    assert 'supporting_facts' not in example['raw_example']
    assert example['evidence_exclusion_reasons'] == ['missing_field:supporting_facts']


def test_test_split_never_eligible_for_correctness_or_evidence(row):
    from src.datasets.two_wiki import evaluation_eligibility
    for key in ('answer', 'answer_id', 'entity_ids'):
        row[key] = ''
    for key in ('supporting_facts', 'evidences', 'evidences_id'):
        row[key] = []
    example = normalize_example(row, split='test', evidence_policy='retain')
    assert not example['qa_valid'] and not example['evidence_valid']
    assert evaluation_eligibility(example)['qa_exclusion_reasons'] == ['unlabeled_official_test']


def test_evidence_exclusion_counts_are_per_question_not_per_bad_fact(row):
    from scripts.validate_two_wiki import validate_split
    bad = deepcopy(row)
    bad['_id'] = 'bad'
    bad['supporting_facts'] = [['Title A', 99], ['Title B', 99]]
    stats = validate_split([row, bad], 'dev', {}, set())
    assert stats['evaluation_eligibility_counts'] == {'qa_valid': 2, 'evidence_valid': 1,
                                                     'qa_excluded': 0, 'evidence_excluded': 1}
    assert stats['evidence_exclusion_question_counts_by_reason'] == {'invalid_sentence_index': 1}
    assert stats['issue_counts']['invalid_sentence_index'] == 2


def test_invalid_policy_rejected(row):
    with pytest.raises(ValueError, match='evidence_policy'):
        normalize_example(row, split='dev', evidence_policy='repair')
