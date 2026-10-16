"""Read-only integrity and complete raw-context provenance verification."""
import argparse
from collections import Counter
import json
from itertools import zip_longest
from pathlib import Path
import sqlite3
import sys
import time
import zipfile
import ijson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.build_pooled_corpus import content_hash, encode, passage_content, peak_rss
from src.datasets.two_wiki import sha256_file


def verify(directory, database, archive_path):
    directory, database, archive_path = map(Path, (directory, database, archive_path))
    start = time.perf_counter()
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    without_checksum = {key: value for key, value in manifest.items() if key != 'corpus_checksum'}
    if content_hash(encode(without_checksum)) != manifest['corpus_checksum']:
        raise ValueError('Manifest corpus checksum mismatch')
    if sha256_file(archive_path) != manifest['dataset_sha256']:
        raise ValueError('Source archive SHA256 mismatch')
    for name, checksum in manifest['output_file_checksums'].items():
        if sha256_file(directory / name) != checksum:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    connection = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
    connection.execute('PRAGMA cache_size=-32768')
    counts = {}
    try:
        for filename, query in [
            ('documents.jsonl', 'SELECT payload FROM documents ORDER BY id'),
            ('source_variants.jsonl', 'SELECT payload FROM variants ORDER BY id'),
            ('source_mappings.jsonl', 'SELECT payload FROM refs ORDER BY doc_id,split,question_id,position'),
        ]:
            count = 0
            with (directory / filename).open('rb') as handle:
                for actual, expected in zip_longest(handle, connection.execute(query)):
                    if actual is None or expected is None or actual != (expected[0] + '\n').encode('utf-8'):
                        raise ValueError(f'Output differs from ordered prepared records: {filename}:{count}')
                    count += 1
            counts[filename] = count
        verified = Counter()
        with zipfile.ZipFile(archive_path) as archive:
            for split in manifest['included_splits']:
                with archive.open(f'{split}.json') as handle:
                    for row in ijson.items(handle, 'item'):
                        for position, entry in enumerate(row['context']):
                            canonical, _, _ = passage_content(*entry)
                            found = connection.execute('''
                                SELECT v.raw,r.doc_id FROM refs r JOIN variants v
                                ON json_extract(r.payload,'$.variant_id')=v.id
                                WHERE r.split=? AND r.question_id=? AND r.position=?
                            ''', (split, row['_id'], position)).fetchone()
                            if found != (encode(entry), '2wiki-v1:' + content_hash(canonical)):
                                raise ValueError(f'Raw context reconstruction mismatch: {split}:{row["_id"]}:{position}')
                            verified[split] += 1
        if sum(verified.values()) != counts['source_mappings.jsonl'] or counts['documents.jsonl'] != manifest['unique_passages']:
            raise ValueError('Manifest or reference count mismatch')
    finally:
        connection.close()
    return {'corpus_checksum': manifest['corpus_checksum'], 'all_file_checksums_match': True,
            'ordered_file_record_counts': counts, 'raw_context_entries_reconstructed_by_split': dict(verified),
            'every_original_context_entry_preserved': True, 'elapsed_seconds': time.perf_counter() - start,
            'peak_process_ram_bytes': peak_rss(), 'raw_archive_sha256_after': sha256_file(archive_path)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus-dir', required=True, type=Path)
    parser.add_argument('--prepared-db', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Report already exists; choose a new path')
    report = verify(args.corpus_dir, args.prepared_db, args.archive)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(encode(report))
