"""Explicit acquisition of reviewed, hash-locked model assets; never loads model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    required = sum(f['size_bytes'] for f in lock['files'].values())
    if shutil.disk_usage(args.output).free < required * 2:
        raise RuntimeError('Insufficient free disk for locked model')
    for name, file in lock['files'].items():
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if digest(target) != file['sha256']:
                raise ValueError(f'Existing file hash mismatch; not overwritten: {target}')
            print(f'Already verified {name}', flush=True)
            continue
        part = target.with_suffix(target.suffix + '.part')
        with urllib.request.urlopen(file['url'], timeout=120) as response, part.open('xb') as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        if part.stat().st_size != file['size_bytes'] or digest(part) != file['sha256']:
            raise ValueError(f'Download hash/size mismatch: {name}; partial file preserved')
        part.rename(target)
        print(f'Verified {name}: {file["size_bytes"]} bytes', flush=True)


if __name__ == '__main__':
    main()
