"""Phase 2 only: streaming ZIP -> disk-backed dedup -> immutable JSONL corpus."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import unicodedata
import zipfile

import ijson
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.two_wiki import sha256_file, validation_flags

BUILDER_VERSION = '2wiki-pooled-v1'
POLICIES = {'train_plus_validation': ('train', 'dev'), 'train_only': ('train',)}


class HashCollisionError(ValueError):
    pass


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def content_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def canonicalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def passage_content(title, sentences):
    if (not isinstance(title, str) or not title.strip() or not isinstance(sentences, list)
            or not sentences or any(not isinstance(s, str) or not s.strip() for s in sentences)):
        raise ValueError('Malformed context title or ordered sentence list; no entries are skipped')
    original_text = ' '.join(sentences)
    canonical = encode([canonicalize(title), canonicalize(original_text)])
    return canonical, original_text, title + '\n' + original_text


def peak_rss():
    memory = psutil.Process().memory_info()
    return getattr(memory, 'peak_wset', memory.rss)


def connect(path):
    connection = sqlite3.connect(path)
    connection.execute('PRAGMA cache_size=-32768')
    connection.execute('PRAGMA temp_store=FILE')
    connection.execute('PRAGMA mmap_size=0')
    return connection


def prepare(archive_path, provenance, work_dir, *, policy='train_plus_validation', hash_fn=content_hash):
    """Exact dry-run using bounded-memory disk state; creates no final corpus."""
    if policy not in POLICIES:
        raise ValueError(f'Unknown corpus policy: {policy}')
    archive_path, work_dir = Path(archive_path), Path(work_dir)
    actual_hash = sha256_file(archive_path)
    if provenance.get('sha256') != actual_hash:
        raise ValueError('Dataset SHA256 does not match source provenance')
    work_dir.mkdir(parents=True, exist_ok=False)
    database = work_dir / 'prepared.sqlite'
    start = time.perf_counter()
    available = shutil.disk_usage(work_dir).free
    with zipfile.ZipFile(archive_path) as archive:
        members = [archive.getinfo(f'{split}.json') for split in POLICIES[policy]]
        conservative_disk = sum(m.file_size for m in members) * 8
        if conservative_disk > available * 0.8 or psutil.virtual_memory().available < 1024**3:
            raise RuntimeError('Unexpected resource pressure: stop for approval before processing')
        connection = connect(database)
        try:
            connection.executescript('''
                CREATE TABLE documents(id TEXT PRIMARY KEY, canonical TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE variants(id TEXT PRIMARY KEY, raw TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE questions(split TEXT, id TEXT, PRIMARY KEY(split,id));
                CREATE TABLE refs(doc_id TEXT, split TEXT, question_id TEXT, position INTEGER,
                                  payload TEXT NOT NULL, PRIMARY KEY(split,question_id,position));
                CREATE TABLE metadata(payload TEXT NOT NULL);
            ''')
            split_counts = {}
            source_metadata = {}
            total = 0
            for split, member in zip(POLICIES[policy], members):
                with archive.open(member) as handle:
                    digest = hashlib.sha256()
                    for block in iter(lambda: handle.read(1024 * 1024), b''):
                        digest.update(block)
                source_metadata[split] = {
                    'dataset': '2wikimultihopqa', 'source_url': provenance.get('source_url'),
                    'revision': provenance.get('version'), 'dataset_sha256': actual_hash,
                    'original_filename': provenance.get('original_filename', archive_path.name),
                    'archive_member': member.filename, 'member_sha256': digest.hexdigest(),
                    'member_size_bytes': member.file_size,
                }
                counts = Counter()
                with archive.open(member) as handle:
                    for row_index, row in enumerate(ijson.items(handle, 'item')):
                        if not isinstance(row, dict) or not isinstance(row.get('_id'), str) or not row['_id']:
                            raise ValueError(f'{split}:{row_index}: missing original question ID')
                        if not isinstance(row.get('context'), list) or not row['context']:
                            raise ValueError(f'{split}:{row_index}: invalid context; build stopped')
                        try:
                            connection.execute('INSERT INTO questions VALUES (?,?)', (split, row['_id']))
                        except sqlite3.IntegrityError as exc:
                            raise ValueError(f'Duplicate question ID in {split}: {row["_id"]}') from exc
                        flags = validation_flags(row, split=split)
                        counts['questions'] += 1
                        counts['qa_valid_questions'] += flags['qa_valid']
                        counts['evidence_invalid_questions'] += not flags['evidence_valid']
                        for position, entry in enumerate(row['context']):
                            if not isinstance(entry, list) or len(entry) != 2:
                                raise ValueError(f'{split}:{row_index}: malformed context entry {position}')
                            title, sentences = entry
                            canonical, text, embedding_text = passage_content(title, sentences)
                            digest = hash_fn(canonical)
                            doc_id = '2wiki-v1:' + digest
                            found = connection.execute('SELECT canonical FROM documents WHERE id=?', (doc_id,)).fetchone()
                            if found and found[0] != canonical:
                                raise HashCollisionError(f'Content hash collision: {digest}; build stopped')
                            raw = encode([title, sentences])
                            variant_id = 'variant-v1:' + content_hash(raw)
                            variant = connection.execute('SELECT raw FROM variants WHERE id=?', (variant_id,)).fetchone()
                            if variant and variant[0] != raw:
                                raise HashCollisionError(f'Raw variant hash collision: {variant_id}')
                            if not variant:
                                payload = {'variant_id': variant_id, 'document_id': doc_id, 'title': title,
                                           'sentences': sentences, 'original_text': text, 'embedding_text': embedding_text}
                                connection.execute('INSERT INTO variants VALUES (?,?,?)', (variant_id, raw, encode(payload)))
                            if not found:
                                payload = {'document_id': doc_id, 'content_hash': digest, 'title': title,
                                           'sentences': sentences, 'original_text': text, 'embedding_text': embedding_text,
                                           'representative_variant_id': variant_id}
                                connection.execute('INSERT INTO documents VALUES (?,?,?)', (doc_id, canonical, encode(payload)))
                            reference = {'document_id': doc_id, 'variant_id': variant_id, 'source_split': split,
                                         'source_question_id': row['_id'], 'context_position': position,
                                         'source_row_index': row_index, 'source_provenance_id': split,
                                         'qa_valid': flags['qa_valid'], 'evidence_valid': flags['evidence_valid'],
                                         'qa_exclusion_reasons': flags['qa_exclusion_reasons'],
                                         'evidence_exclusion_reasons': flags['evidence_exclusion_reasons']}
                            connection.execute('INSERT INTO refs VALUES (?,?,?,?,?)',
                                               (doc_id, split, row['_id'], position, encode(reference)))
                            counts['context_entries'] += 1
                            counts['evidence_invalid_context_entries_included'] += not flags['evidence_valid']
                            total += 1
                        if counts['questions'] % 2000 == 0:
                            connection.commit()
                connection.commit()
                split_counts[split] = dict(counts)
                print(encode({'prepared_split': split, **dict(counts)}), flush=True)
            connection.execute('CREATE INDEX refs_doc_split ON refs(doc_id,split,question_id,position)')
            unique = connection.execute('SELECT count(*) FROM documents').fetchone()[0]
            per_split = {split: connection.execute('SELECT count(DISTINCT doc_id) FROM refs WHERE split=?', (split,)).fetchone()[0]
                         for split in POLICIES[policy]}
            shared = connection.execute('SELECT count(*) FROM (SELECT doc_id FROM refs GROUP BY doc_id HAVING count(DISTINCT split)>1)').fetchone()[0]
            cross_entries = connection.execute('SELECT count(*) FROM refs WHERE doc_id IN (SELECT doc_id FROM refs GROUP BY doc_id HAVING count(DISTINCT split)>1)').fetchone()[0]
            stable = {
                'schema_version': 1, 'builder_version': BUILDER_VERSION, 'dataset': '2wikimultihopqa',
                'dataset_source_url': provenance.get('source_url'), 'dataset_revision': provenance.get('version'),
                'dataset_sha256': actual_hash, 'included_splits': list(POLICIES[policy]), 'corpus_policy': policy,
                'benchmark': 'pooled-context adaptation; dev contexts included only under train_plus_validation',
                'passage_granularity': 'one original context entry; no chunking',
                'embedding_text_rule': 'original title + LF + space.join(original ordered sentences)',
                'canonicalization': 'NFC then Unicode whitespace split/join independently on complete title and joined passage; canonical JSON [title,text]',
                'document_id_algorithm': '2wiki-v1: + SHA256(UTF8 canonical JSON [title,text]); compare canonical bytes on hash match, abort collisions',
                'raw_variant_algorithm': 'variant-v1: + SHA256(UTF8 canonical JSON [original title,original sentence list])',
                'representative_rule': 'first occurrence in fixed train/dev order, original row order and context position; variants preserve every alternative',
                'serialization': 'UTF-8, ensure_ascii=false, sorted keys, compact separators, LF; documents/variants sorted by ID; refs by doc_id,split,question_id,position',
                'data_quality_policy': 'all context entries included regardless of QA/evidence flags; no gold/support-based selection',
                'total_context_entries': total, 'unique_passages': unique, 'duplicate_count': total - unique,
                'unique_raw_variants': connection.execute('SELECT count(*) FROM variants').fetchone()[0],
                'split_statistics': split_counts,
                'cross_split_overlap': {'unique_passages_per_split': per_split, 'shared_unique_passages': shared,
                                       'source_entries_on_shared_passages': cross_entries,
                                       'definition': 'shared = canonical documents appearing in >1 included split; entry count includes all occurrences'},
            }
            connection.execute('INSERT INTO metadata VALUES (?)', (encode({'manifest': stable, 'provenance': source_metadata}),))
            connection.commit()
            byte_counts = {filename: connection.execute(f'SELECT coalesce(sum(length(cast(payload AS BLOB))+1),0) FROM {table}').fetchone()[0]
                           for filename, table in [('documents.jsonl', 'documents'), ('source_variants.jsonl', 'variants'), ('source_mappings.jsonl', 'refs')]}
            byte_counts['provenance.json'] = len((encode(source_metadata) + '\n').encode('utf-8'))
        finally:
            connection.close()
    elapsed = time.perf_counter() - start
    return {'mode': 'dry-run', 'prepared_database': str(database.resolve()),
            'prepared_database_sha256': sha256_file(database), 'manifest_preview': stable,
            'estimated_output_bytes_by_file': byte_counts,
            'estimated_output_bytes_excluding_manifest': sum(byte_counts.values()),
            'available_disk_bytes_before': available, 'available_disk_bytes_after': shutil.disk_usage(work_dir).free,
            'sqlite_bytes': database.stat().st_size, 'observed_peak_process_ram_bytes': peak_rss(),
            'estimated_peak_ram_bytes': max(peak_rss() * 2, 256 * 1024**2),
            'dry_run_seconds': elapsed, 'estimated_export_seconds_at_20_MiB_s': sum(byte_counts.values()) / (20 * 1024**2),
            'runtime_estimate_note': 'Export estimate assumes 20 MiB/s sequential throughput; sorted SQLite scan overhead and antivirus may add time.'}


def export_prepared(database, output_root, *, expected_database_sha256):
    """Reuse audited dry-run state; publish only complete, checksummed assets."""
    database, output_root = Path(database), Path(output_root)
    if sha256_file(database) != expected_database_sha256:
        raise ValueError('Prepared database differs from dry-run checksum')
    output_root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    connection = connect(database)
    staging = output_root / ('.building-' + expected_database_sha256[:16])
    if staging.exists():
        raise FileExistsError(f'Existing staging directory is preserved: {staging}')
    try:
        metadata = json.loads(connection.execute('SELECT payload FROM metadata').fetchone()[0])
        manifest = metadata['manifest']
        estimated = sum(connection.execute(f'SELECT coalesce(sum(length(cast(payload AS BLOB))+1),0) FROM {table}').fetchone()[0]
                        for table in ('documents', 'variants', 'refs'))
        if estimated * 2 > shutil.disk_usage(output_root).free:
            raise RuntimeError('Insufficient safe disk margin; stop for approval')
        staging.mkdir()
        checksums = {}
        sizes = {}
        for filename, query in [
            ('documents.jsonl', 'SELECT payload FROM documents ORDER BY id'),
            ('source_variants.jsonl', 'SELECT payload FROM variants ORDER BY id'),
            ('source_mappings.jsonl', 'SELECT payload FROM refs ORDER BY doc_id,split,question_id,position'),
        ]:
            digest = hashlib.sha256()
            with (staging / filename).open('xb') as handle:
                for (payload,) in connection.execute(query):
                    encoded = (payload + '\n').encode('utf-8')
                    handle.write(encoded)
                    digest.update(encoded)
            checksums[filename] = digest.hexdigest()
            sizes[filename] = (staging / filename).stat().st_size
        (staging / 'provenance.json').write_bytes((encode(metadata['provenance']) + '\n').encode('utf-8'))
        checksums['provenance.json'] = sha256_file(staging / 'provenance.json')
        sizes['provenance.json'] = (staging / 'provenance.json').stat().st_size
        manifest['output_file_checksums'] = checksums
        manifest['output_file_sizes_bytes'] = sizes
        checksum = content_hash(encode(manifest))
        manifest['corpus_checksum'] = checksum
        (staging / 'manifest.json').write_bytes((encode(manifest) + '\n').encode('utf-8'))
        target = output_root / checksum
        if target.exists():
            # Never overwrite a previous run; verify exact identity instead.
            for filename in [*checksums, 'manifest.json']:
                if sha256_file(target / filename) != sha256_file(staging / filename):
                    raise ValueError(f'Existing corpus differs: {filename}; both copies preserved')
            published = False
        else:
            staging.rename(target)
            published = True
        return {'output_directory': str(target.resolve()), 'corpus_checksum': checksum,
                'manifest': manifest, 'export_seconds': time.perf_counter() - start,
                'peak_process_ram_bytes': peak_rss(), 'newly_published': published,
                'identical_rebuild_staging_preserved': None if published else str(staging.resolve())}
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['dry-run', 'build'], required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--policy', choices=list(POLICIES), default='train_plus_validation')
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--dry-run-report', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path('artifacts/2wiki'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Report already exists; choose a new path')
    provenance = json.loads(args.provenance.read_text(encoding='utf-8'))
    if args.mode == 'dry-run':
        if args.work_dir is None:
            parser.error('--work-dir is required for dry-run')
        result = prepare(args.archive, provenance, args.work_dir, policy=args.policy)
    else:
        if args.dry_run_report is None:
            parser.error('--dry-run-report is required before building')
        dry_run = json.loads(args.dry_run_report.read_text(encoding='utf-8'))
        if sha256_file(args.archive) != dry_run['manifest_preview']['dataset_sha256']:
            raise ValueError('Raw archive changed after dry-run')
        if args.policy != dry_run['manifest_preview']['corpus_policy']:
            raise ValueError('Policy differs from dry-run')
        result = export_prepared(dry_run['prepared_database'], args.output_root,
                                 expected_database_sha256=dry_run['prepared_database_sha256'])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(encode(result), flush=True)


if __name__ == '__main__':
    main()
