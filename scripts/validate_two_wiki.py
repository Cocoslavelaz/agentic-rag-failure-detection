"""Exhaustive read-only schema/reference audit; errors are reported, never repaired."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.two_wiki import validation_flags, read_source


def validate_split(rows, split, source, global_ids):
    types, contexts, facts_counts = Counter(), Counter(), Counter()
    issues, missing, nulls, empty, fields = Counter(), Counter(), Counter(), Counter(), Counter()
    seen = set()
    duplicate_ids = cross_duplicates = invalid_rows = total_facts = matched_facts = 0
    cases = []
    eligibility, qa_exclusions, evidence_exclusions = Counter(), Counter(), Counter()
    keys = sorted({key for row in rows if isinstance(row, dict) for key in row})
    for index, row in enumerate(rows):
        flags = validation_flags(row, split=split)
        errors = flags['validation_issues']
        eligibility['qa_valid'] += flags['qa_valid']
        eligibility['evidence_valid'] += flags['evidence_valid']
        eligibility['qa_excluded'] += not flags['qa_valid']
        eligibility['evidence_excluded'] += not flags['evidence_valid']
        qa_exclusions.update({r.split(':', 1)[0] for r in flags['qa_exclusion_reasons']})
        evidence_exclusions.update({r.split(':', 1)[0] for r in flags['evidence_exclusion_reasons']})
        if isinstance(row, dict):
            identifier = row.get('_id')
            if isinstance(identifier, str):
                if identifier in seen:
                    duplicate_ids += 1
                    errors.append({'code': 'duplicate_question_id', 'detail': identifier})
                if identifier in global_ids:
                    cross_duplicates += 1
                    errors.append({'code': 'cross_split_duplicate_id', 'detail': identifier})
                seen.add(identifier)
            types[str(row.get('type'))] += 1
            fields[tuple(sorted(row))] += 1
            for key in keys:
                if key not in row:
                    missing[key] += 1
                elif row[key] is None:
                    nulls[key] += 1
                elif row[key] in ('', []):
                    empty[key] += 1
            context = row.get('context')
            facts = row.get('supporting_facts')
            contexts[len(context) if isinstance(context, list) else 'invalid'] += 1
            facts_counts[len(facts) if isinstance(facts, list) else 'invalid'] += 1
            if isinstance(facts, list):
                for fact in facts:
                    total_facts += 1
                    if not isinstance(fact, list) or len(fact) != 2 or type(fact[1]) is not int:
                        continue
                    matches = [entry[1] for entry in (context if isinstance(context, list) else []) if isinstance(entry, list) and len(entry) == 2
                               and entry[0] == fact[0] and isinstance(entry[1], list)]
                    if len(matches) == 1 and 0 <= fact[1] < len(matches[0]):
                        matched_facts += 1
        if errors:
            invalid_rows += 1
            issues.update(error['code'] for error in errors)
            if len(cases) < 20:
                cases.append({'row_index': index, 'question_id': row.get('_id') if isinstance(row, dict) else None,
                              'issues': errors,
                              'supporting_facts': row.get('supporting_facts') if isinstance(row, dict) else None,
                              'context_sentence_counts': [[e[0], len(e[1])] for e in row.get('context', [])
                                                          if isinstance(e, list) and len(e) == 2 and isinstance(e[1], list)]
                              if isinstance(row, dict) and isinstance(row.get('context'), list) else None})
    global_ids.update(seen)
    return {
        'source_provenance': source, 'sample_count': len(rows), 'unique_question_ids': len(seen),
        'duplicate_id_occurrences': duplicate_ids, 'cross_split_duplicate_occurrences': cross_duplicates,
        'question_type_distribution': dict(types),
        'context_entry_count_distribution': dict(contexts), 'supporting_fact_count_distribution': dict(facts_counts),
        'supporting_fact_mapping': {'total': total_facts, 'matched': matched_facts,
                                    'unmatched': total_facts - matched_facts,
                                    'success_rate': matched_facts / total_facts if total_facts else None},
        'field_sets': [{'fields': list(k), 'count': v} for k, v in fields.items()],
        'missing_field_counts': dict(missing), 'null_field_counts': dict(nulls),
        'empty_field_counts': dict(empty), 'invalid_example_count': invalid_rows,
        'issue_counts': dict(issues), 'issue_examples_first_20': cases,
        'correctness_evaluation_available': split != 'test',
        'evaluation_eligibility_counts': dict(eligibility),
        'qa_exclusion_question_counts_by_reason': dict(qa_exclusions),
        'evidence_exclusion_question_counts_by_reason': dict(evidence_exclusions),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--provenance', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new report path')
    provenance = json.loads(args.provenance.read_text(encoding='utf-8'))
    report = {'dataset': '2wikimultihopqa', 'validated_at': datetime.now(timezone.utc).isoformat(),
              'no_records_skipped': True, 'source': provenance, 'splits': {}}
    all_ids = set()
    for split in ('train', 'dev', 'test'):
        rows, source = read_source(args.archive, split=split, provenance=provenance)
        if source['source_sha256'] != provenance['sha256']:
            raise ValueError('Source archive SHA256 does not match provenance')
        stats = validate_split(rows, split, source, all_ids)
        report['splits'][split] = stats
        print(json.dumps({'split': split, 'sample_count': stats['sample_count'],
                          'invalid_example_count': stats['invalid_example_count'],
                          'issue_counts': stats['issue_counts']}, ensure_ascii=False), flush=True)
        del rows
    report['valid'] = all(s['invalid_example_count'] == 0 for s in report['splits'].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
