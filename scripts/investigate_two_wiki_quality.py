"""Independent raw-ZIP fact audit plus explicit evaluation-exclusion accounting."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.two_wiki import (OFFICIAL_ARCHIVE_SHA256, normalize_example,
                                   sha256_file, validation_flags)


def text_hash(sentences):
    return hashlib.sha256(' '.join(' '.join(sentences).split()).encode('utf-8')).hexdigest()


def raw_facts(row):
    """Classify without calling adapter, preserving each original index."""
    invalid = []
    context = row.get('context')
    facts = row.get('supporting_facts')
    if not isinstance(facts, list):
        return [{'cause': 'other', 'detail': 'malformed supporting_facts'}]
    for position, fact in enumerate(facts):
        result = {'fact_position': position, 'original_annotation': fact}
        if not isinstance(fact, list) or len(fact) != 2 or not isinstance(fact[0], str) or type(fact[1]) is not int:
            invalid.append({**result, 'cause': 'other', 'detail': 'malformed supporting fact'})
            continue
        matches = [entry for entry in context if isinstance(entry, list) and len(entry) == 2 and entry[0] == fact[0]] if isinstance(context, list) else []
        if not matches:
            invalid.append({**result, 'cause': 'missing_context_title'})
        elif len(matches) != 1:
            invalid.append({**result, 'cause': 'other', 'detail': 'ambiguous context title'})
        elif not isinstance(matches[0][1], list) or not matches[0][1]:
            invalid.append({**result, 'cause': 'empty_sentence_list'})
        elif not 0 <= fact[1] < len(matches[0][1]):
            invalid.append({**result, 'cause': 'sentence_index_out_of_range',
                            'sentence_count': len(matches[0][1]), 'index': fact[1],
                            'normalized_text_sha256': text_hash(matches[0][1]),
                            'last_sentence': matches[0][1][-1]})
    return invalid


def investigate(archive_path):
    before = sha256_file(archive_path)
    if before != OFFICIAL_ARCHIVE_SHA256:
        raise ValueError('Archive differs from audited official corrected release')
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'archive_sha256': before,
              'version': '2021-04-07',
              'official_documentation': 'https://github.com/Alab-NII/2wikimultihop#new-update-april-7-2021',
              'method': 'Direct zipfile + json.load independent of dataset adapter; all rows inspected, no repair/filtering',
              'splits': {}, 'invalid_facts': [], 'representative_examples': []}
    targets = set()
    with zipfile.ZipFile(archive_path) as archive:
        for split in ('train', 'dev', 'test'):
            with archive.open(f'{split}.json') as handle:
                rows = json.load(handle)
            affected_types, fact_types, causes, qa_reasons, evidence_reasons = [Counter() for _ in range(5)]
            counts = Counter()
            for index, row in enumerate(rows):
                flags = validation_flags(row, split=split)
                counts['qa_valid'] += flags['qa_valid']
                counts['evidence_valid'] += flags['evidence_valid']
                counts['qa_excluded'] += not flags['qa_valid']
                counts['evidence_excluded'] += not flags['evidence_valid']
                qa_reasons.update({r.split(':', 1)[0] for r in flags['qa_exclusion_reasons']})
                evidence_reasons.update({r.split(':', 1)[0] for r in flags['evidence_exclusion_reasons']})
                bad = raw_facts(row)
                if not bad:
                    continue
                counts['affected_questions'] += 1
                counts['invalid_supporting_facts'] += len(bad)
                affected_types[row['type']] += 1
                fact_types[row['type']] += len(bad)
                normalized = normalize_example(row, split=split, evidence_policy='retain', row_index=index)
                preserved = (normalized['raw_example'] == row and
                             normalized['supporting_facts'] == row['supporting_facts'] and
                             [[e['title'], e['sentences']] for e in normalized['context_entries']] == row['context'] and
                             normalized['gold_answer'] == row['answer'])
                counts['adapter_preservation_verified_questions'] += preserved
                counts['gold_answer_present_questions'] += isinstance(row.get('answer'), str) and bool(row['answer'].strip())
                counts['context_present_questions'] += bool(row.get('context'))
                literal = row['answer'].casefold() in ' '.join(s for _, sentences in row['context'] for s in sentences).casefold()
                counts['literal_gold_answer_in_context_questions'] += literal
                for fact in bad:
                    causes[fact['cause']] += 1
                    item = {'split': split, 'row_index': index, 'question_id': row['_id'],
                            'question_type': row['type'], **fact}
                    report['invalid_facts'].append(item)
                    if fact['cause'] == 'sentence_index_out_of_range':
                        targets.add(fact['original_annotation'][0])
                if affected_types[row['type']] <= 2:
                    report['representative_examples'].append({
                        'split': split, 'row_index': index, 'question_id': row['_id'],
                        'question_type': row['type'], 'question': row['question'], 'gold_answer': row['answer'],
                        'context_entry_count': len(row['context']), 'invalid_facts': bad,
                        'original_supporting_facts': row['supporting_facts'],
                        'adapter_preserves_raw_and_annotations': preserved,
                        'gold_answer_literal_in_context': literal,
                        'qa_valid': flags['qa_valid'], 'evidence_valid': flags['evidence_valid']})
            report['splits'][split] = {'sample_count': len(rows), **dict(counts),
                                       'affected_questions_by_type': dict(affected_types),
                                       'invalid_facts_by_type': dict(fact_types), 'proven_fact_causes': dict(causes),
                                       'qa_exclusion_question_counts_by_reason': dict(qa_reasons),
                                       'evidence_exclusion_question_counts_by_reason': dict(evidence_reasons)}
            print(json.dumps({'split': split, **dict(counts)}, ensure_ascii=False), flush=True)
            del rows
        # Search the same release for equal normalized article text with a
        # different sentence count. This is evidence of boundary variation,
        # never a license to map an invalid annotation to a new sentence.
        variants = defaultdict(dict)
        for split in ('train', 'dev', 'test'):
            with archive.open(f'{split}.json') as handle:
                rows = json.load(handle)
            for index, row in enumerate(rows):
                for title, sentences in row['context']:
                    if title in targets:
                        key = (title, text_hash(sentences))
                        variants[key].setdefault(len(sentences), {'split': split, 'row_index': index,
                                                                 'question_id': row['_id']})
            del rows
    variation = Counter()
    for fact in report['invalid_facts']:
        if fact['cause'] != 'sentence_index_out_of_range':
            continue
        alternatives = variants[(fact['original_annotation'][0], fact['normalized_text_sha256'])]
        larger = {str(n): sample for n, sample in alternatives.items() if n > fact['index']}
        fact['same_text_alternative_sentence_counts'] = sorted(alternatives)
        fact['same_text_index_in_range_alternatives'] = larger
        variation['facts_with_same_text_different_sentence_counts'] += len(alternatives) > 1
        variation['facts_with_same_text_index_in_range_elsewhere'] += bool(larger)
    report['sentence_segmentation_investigation'] = {
        **dict(variation), 'confirmed_historical_segmentation_root_cause_count': None,
        'interpretation': 'Index inconsistency is proven in raw corrected-release JSON. Same-text boundary variation is diagnostic only; historical annotation-generation alignment cannot be established without author provenance. No old release or external corpus downloaded.'}
    report['raw_archive_unchanged'] = sha256_file(archive_path) == before
    report['adapter_introduced_issue_questions'] = sum(
        s.get('affected_questions', 0) - s.get('adapter_preservation_verified_questions', 0)
        for s in report['splits'].values())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new path')
    result = investigate(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
