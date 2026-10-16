"""FAISS process boundary prevents conflicting Windows OpenMP runtimes."""
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import subprocess
import sys
from src.embedding_contract import stable_json, validate_vectors


def _request(operation, **kwargs):
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [sys.executable, str(root / 'scripts/faiss_index_worker.py')],
        input=stable_json({'operation': operation, **kwargs}),
        text=True, encoding='utf-8', capture_output=True, cwd=root,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'FAISS worker failed ({process.returncode}): {process.stderr}') from exc
    if process.returncode or isinstance(result, dict) and 'worker_error' in result:
        raise ValueError(f'FAISS worker rejected request: {result}; {process.stderr}')
    return result


def build_index(output_dir, documents, embeddings, **kwargs):
    config = kwargs['encoder_config']
    config = asdict(config) if is_dataclass(config) else config
    validate_vectors(embeddings, config['dimension'])
    kwargs['encoder_config'] = config
    return _request('build', output_dir=str(output_dir), documents=documents,
                    embeddings=embeddings.tolist(), **kwargs)


class ProcessFaissIndex:
    @classmethod
    def load(cls, directory, **kwargs):
        config = kwargs['expected_encoder_config']
        kwargs['expected_encoder_config'] = asdict(config) if is_dataclass(config) else config
        result = cls()
        result.request = {'directory': str(directory), **kwargs}
        loaded = _request('load', **result.request)
        result.manifest, result.documents = loaded['manifest'], loaded['documents']
        return result

    def search(self, queries, *, k=10):
        validate_vectors(queries, self.manifest['embedding_dimension'])
        return _request('search', vectors=queries.tolist(), k=k, **self.request)

    def compare_cosine(self, queries, *, k=10):
        validate_vectors(queries, self.manifest['embedding_dimension'])
        return _request('cosine', vectors=queries.tolist(), k=k, **self.request)
