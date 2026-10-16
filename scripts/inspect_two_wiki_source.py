"""Inspect official archive without extracting or changing its original bytes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(archive):
    result = {'archive': str(archive), 'size_bytes': archive.stat().st_size,
              'sha256': sha256_file(archive), 'inspected_at': datetime.now(timezone.utc).isoformat(),
              'members': [], 'splits': {}}
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            result['members'].append({'name': info.filename, 'size_bytes': info.file_size})
            if Path(info.filename).name not in {'train.json', 'dev.json', 'test.json'}:
                continue
            with zipped.open(info) as handle:
                rows = json.load(handle)
            split = Path(info.filename).stem
            fields = Counter(tuple(sorted(row)) for row in rows)
            result['splits'][split] = {
                'count': len(rows), 'field_sets': [{'fields': list(keys), 'count': count} for keys, count in fields.items()],
                'first_example': rows[0],
                'field_types': {key: dict(Counter(type(row[key]).__name__ for row in rows if key in row))
                                for key in sorted({key for row in rows for key in row})}}
            del rows
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != 'splits'}, ensure_ascii=False))
    print(json.dumps({split: {key: value for key, value in stats.items() if key != 'first_example'}
                      for split, stats in report['splits'].items()}, ensure_ascii=False))
