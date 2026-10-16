"""Download small primary-source metadata/code for review, never execute it."""
import json
from pathlib import Path
import urllib.request


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def main():
    output = Path('outputs/phase3a/model_review')
    output.mkdir(parents=True, exist_ok=False)
    model_id = 'Alibaba-NLP/gte-multilingual-base'
    info = json.loads(fetch(f'https://huggingface.co/api/models/{model_id}?blobs=true'))
    (output / 'model_api.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    revision = info['sha']
    for name in ('config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'README.md'):
        (output / name).write_bytes(fetch(f'https://huggingface.co/{model_id}/resolve/{revision}/{name}'))
    config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    code_repos = {value.split('--')[0] for value in config.get('auto_map', {}).values()
                  if isinstance(value, str) and '--' in value}
    for repo in code_repos:
        code_info = json.loads(fetch(f'https://huggingface.co/api/models/{repo}?blobs=true'))
        code_dir = output / repo.split('/')[-1]
        code_dir.mkdir()
        (code_dir / 'api.json').write_text(json.dumps(code_info, indent=2), encoding='utf-8')
        for item in code_info['siblings']:
            name = item['rfilename']
            if name.endswith('.py'):
                target = code_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(fetch(f'https://huggingface.co/{repo}/resolve/{code_info["sha"]}/{name}'))
    print(json.dumps({'model_id': model_id, 'revision': revision, 'config': config,
                      'files': [{k: v for k, v in item.items() if k in {'rfilename', 'size', 'lfs'}} for item in info['siblings']],
                      'code_repositories': sorted(code_repos)}, indent=2))


if __name__ == '__main__':
    main()
