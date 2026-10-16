"""Strict adapter for the author's 2021-04-07 2Wiki release. No model calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

QUESTION_TYPES = {'comparison', 'inference', 'compositional', 'bridge_comparison'}
SOURCE_URL = 'https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1'
SOURCE_VERSION = '2021-04-07'
OFFICIAL_ARCHIVE_SHA256 = '95df2bf56fdabe034e27aebc580e02264232203cf52552f9efe8a919e5529eef'


class DatasetValidationError(ValueError):
    def __init__(self, issues, *, question_id=None, row_index=None):
        self.issues = issues
        self.question_id = question_id
        self.row_index = row_index
        super().__init__(f'2Wiki row={row_index} question_id={question_id!r}: ' +
                         '; '.join('{}: {}'.format(i['code'], i['detail']) for i in issues))


def canonical_split(split):
    if split == 'validation':
        return 'dev'
    if split not in {'train', 'dev', 'test'}:
        raise ValueError('Explicit split must be train, dev/validation or test')
    return split


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_example(row, *, split):
    """Return every schema/reference issue; never repair or drop a record."""
    split = canonical_split(split)
    issues = []
    def issue(code, detail):
        issues.append({'code': code, 'detail': detail})
    def nonempty(value):
        return isinstance(value, str) and bool(value.strip())
    if not isinstance(row, dict):
        return [{'code': 'record_type', 'detail': 'Record must be an object'}]
    required = {'_id', 'question', 'type', 'context'}
    if split != 'test':
        required |= {'answer', 'supporting_facts', 'evidences', 'entity_ids'}
    for key in sorted(required - row.keys()):
        issue('missing_field', key)
    for key in ('_id', 'question', 'type'):
        if key in row and not nonempty(row[key]):
            issue('invalid_string', key)
    if isinstance(row.get('type'), str) and row['type'] not in QUESTION_TYPES:
        issue('unknown_question_type', row['type'])
    if split != 'test' and 'answer' in row and not nonempty(row['answer']):
        issue('invalid_answer', 'Labeled split requires nonempty string answer')
    if 'entity_ids' in row and not isinstance(row['entity_ids'], str):
        issue('invalid_entity_ids', 'Expected string')
    if split != 'test' and row.get('entity_ids') == '':
        issue('invalid_entity_ids', 'Labeled split requires nonempty entity_ids')
    titles = {}
    context = row.get('context')
    if not isinstance(context, list) or not context:
        issue('invalid_context', 'Expected nonempty list')
    else:
        for index, entry in enumerate(context):
            if not isinstance(entry, list) or len(entry) != 2 or not nonempty(entry[0]):
                issue('malformed_context_entry', f'context[{index}]')
                continue
            title, sentences = entry
            if not isinstance(sentences, list) or not sentences or any(not nonempty(s) for s in sentences):
                issue('invalid_sentences', f'context[{index}] {title!r}')
                continue
            titles.setdefault(title, []).append(sentences)
    facts = row.get('supporting_facts', [])
    if not isinstance(facts, list):
        issue('malformed_supporting_facts', 'Expected list')
    else:
        if split != 'test' and 'supporting_facts' in row and not facts:
            issue('empty_supporting_facts', 'Labeled split has no supporting facts')
        for index, fact in enumerate(facts):
            if (not isinstance(fact, list) or len(fact) != 2 or not nonempty(fact[0])
                    or type(fact[1]) is not int):
                issue('malformed_supporting_fact', f'supporting_facts[{index}]={fact!r}')
                continue
            title, sentence_index = fact
            if title not in titles:
                issue('support_title_missing', f'{title!r} at fact {index}')
            elif len(titles[title]) != 1:
                issue('support_title_ambiguous', f'{title!r} at fact {index}')
            elif not 0 <= sentence_index < len(titles[title][0]):
                issue('invalid_sentence_index', f'{title!r} index={sentence_index}, count={len(titles[title][0])}')
    for key in ('evidences', 'evidences_id'):
        if key not in row:
            continue
        triples = row[key]
        if not isinstance(triples, list) or any(not isinstance(t, list) or len(t) != 3 or
                                               any(not nonempty(s) for s in t) for t in triples):
            issue('malformed_evidence', key)
    if 'answer_id' in row and row['answer_id'] is not None and not isinstance(row['answer_id'], str):
        issue('invalid_answer_id', 'Expected string or null (optional alias annotation)')
    if split == 'test':
        for key in ('answer', 'answer_id', 'entity_ids'):
            if key in row and row[key] != '':
                issue('unexpected_test_annotation', key)
        for key in ('supporting_facts', 'evidences', 'evidences_id'):
            if key in row and row[key] != []:
                issue('unexpected_test_annotation', key)
    return issues


def validation_flags(row, *, split):
    """Independent QA/evidence eligibility, never inferred answer correctness."""
    split = canonical_split(split)
    issues = inspect_example(row, split=split)
    annotation_fields = {'supporting_facts', 'evidences', 'entity_ids'}
    annotation_codes = {'malformed_supporting_facts', 'empty_supporting_facts',
                        'malformed_supporting_fact', 'support_title_missing',
                        'support_title_ambiguous', 'invalid_sentence_index',
                        'malformed_evidence', 'invalid_entity_ids', 'invalid_answer_id'}
    context_codes = {'record_type', 'invalid_context', 'malformed_context_entry', 'invalid_sentences'}
    qa_issues, evidence_issues = [], []
    for issue in issues:
        annotation = issue['code'] in annotation_codes or (
            issue['code'] == 'missing_field' and issue['detail'] in annotation_fields)
        if not annotation:
            qa_issues.append(issue)
        if annotation or issue['code'] in context_codes or (issue['code'] == 'invalid_string' and issue['detail'] == '_id') or (
            issue['code'] == 'missing_field' and issue['detail'] in {'_id', 'context'}):
            evidence_issues.append(issue)
    qa_reasons = sorted({i['code'] + ':' + i['detail'] for i in qa_issues})
    evidence_reasons = sorted({i['code'] + ':' + i['detail'] for i in evidence_issues})
    if split == 'test':
        qa_reasons.append('unlabeled_official_test')
        evidence_reasons.append('unlabeled_official_test')
    return {'qa_valid': not qa_reasons, 'evidence_valid': not evidence_reasons,
            'qa_exclusion_reasons': qa_reasons, 'evidence_exclusion_reasons': evidence_reasons,
            'validation_issues': deepcopy(issues)}


def evaluation_eligibility(example):
    """Policy only; no EM/F1 computation and no label-dependent feature values."""
    return {'qa_metrics_eligible': example['qa_valid'],
            'failure_prediction_eligible': example['qa_valid'],
            'annotation_dependent_evidence_metrics_eligible': example['evidence_valid'],
            'qa_exclusion_reasons': list(example['qa_exclusion_reasons']),
            'evidence_exclusion_reasons': list(example['evidence_exclusion_reasons'])}


def normalize_example(row, *, split, provenance=None, row_index=None, evidence_policy='strict'):
    split = canonical_split(split)
    if evidence_policy not in {'strict', 'retain'}:
        raise ValueError('evidence_policy must be strict or retain')
    issues = inspect_example(row, split=split)
    flags = validation_flags(row, split=split)
    # Retention is explicit and only permits annotation issues. Broken QA rows
    # still fail, as do mislabeled official-test annotations.
    if issues and (evidence_policy == 'strict' or not flags['qa_valid']):
        raise DatasetValidationError(issues, question_id=row.get('_id') if isinstance(row, dict) else None,
                                     row_index=row_index)
    return {
        'question_id': row['_id'], 'question': row['question'],
        'gold_answer': None if split == 'test' else row['answer'],
        'question_type': row['type'], 'source_split': split,
        'context_entries': [{'title': title, 'sentences': list(sentences)} for title, sentences in row['context']],
        'supporting_facts': deepcopy(row.get('supporting_facts', [])),
        'evidence_annotations': deepcopy(row.get('evidences', [])),
        'entity_ids': row.get('entity_ids'), 'raw_example': deepcopy(row),
        'correctness_evaluation_available': flags['qa_valid'],
        **flags,
        'evidence_policy': evidence_policy,
        'answerability': 'unknown' if split == 'test' else 'answerable',
        'source_provenance': {**deepcopy(provenance or {}), 'row_index': row_index, 'source_split': split},
    }


def read_source(path=None, *, split, provenance=None):
    """Return raw rows and byte-level provenance. ZIP split selection is explicit."""
    split = canonical_split(split)
    from src.dataset_io import resolve_dataset_path, load_dataset
    source = resolve_dataset_path('2wikimultihopqa', path)
    metadata = {**(provenance or {}), 'dataset': '2wikimultihopqa',
                'source_file': str(source), 'original_filename': source.name,
                'source_sha256': sha256_file(source), 'size_bytes': source.stat().st_size}
    if metadata['source_sha256'] == OFFICIAL_ARCHIVE_SHA256:
        metadata.update(source_url=SOURCE_URL, version=SOURCE_VERSION,
                        source_identity='matches_locally_audited_official_archive')
    else:
        metadata.setdefault('source_identity', 'local_source_not_official_archive_verified')
    if source.suffix.lower() == '.zip':
        with zipfile.ZipFile(source) as archive:
            matches = [i for i in archive.infolist() if Path(i.filename).name == f'{split}.json']
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one {split}.json in archive; found {len(matches)}')
            member = matches[0]
            with archive.open(member) as handle:
                digest = hashlib.sha256()
                # Hash original uncompressed bytes, without reserializing.
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            with archive.open(member) as handle:
                rows = json.load(handle)
            metadata.update(archive_member=member.filename, member_sha256=digest.hexdigest(),
                            member_size_bytes=member.file_size)
    else:
        rows = load_dataset('local', source)
    if not isinstance(rows, list):
        raise ValueError('2Wiki source must contain an array of records')
    return rows, metadata


def iter_two_wiki(path=None, *, split, provenance=None, evidence_policy='strict'):
    rows, metadata = read_source(path, split=split, provenance=provenance)
    seen = set()
    for index, row in enumerate(rows):
        example = normalize_example(row, split=split, provenance=metadata, row_index=index,
                                    evidence_policy=evidence_policy)
        identifier = example['question_id']
        if identifier in seen:
            raise DatasetValidationError([{'code': 'duplicate_question_id', 'detail': identifier}],
                                         question_id=identifier, row_index=index)
        seen.add(identifier)
        yield example


def load_two_wiki(path=None, *, split, provenance=None, evidence_policy='strict'):
    return list(iter_two_wiki(path, split=split, provenance=provenance, evidence_policy=evidence_policy))
