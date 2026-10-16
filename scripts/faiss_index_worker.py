"""One-shot FAISS worker: never imports Torch or executes a model."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import faiss
import numpy as np
from corpus.build_local_index import LocalFaissIndex, build_index, compare_cosine
from src.embedding_contract import stable_json


def main():
    # Windows pipes otherwise inherit the legacy console encoding.
    sys.stdin.reconfigure(encoding='utf-8', errors='strict')
    sys.stdout.reconfigure(encoding='utf-8', errors='strict')
    sys.stderr.reconfigure(encoding='utf-8', errors='strict')
    request = json.load(sys.stdin)
    faiss.omp_set_num_threads(request.pop('threads', 4))
    operation = request.pop('operation')
    if operation == 'build':
        request['embeddings'] = np.asarray(request['embeddings'], dtype=np.float32)
        result = build_index(**request)
    else:
        vectors = request.pop('vectors', None)
        k = request.pop('k', 10)
        index = LocalFaissIndex.load(**request)
        if operation == 'load':
            result = {'manifest': index.manifest, 'documents': index.documents}
        elif operation == 'search':
            result = index.search(np.asarray(vectors, dtype=np.float32), k=k)
        elif operation == 'cosine':
            result = index.compare_cosine(np.asarray(vectors, dtype=np.float32), k=k)
        else:
            raise ValueError('Unknown worker operation')
    print(stable_json(result))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(stable_json({'worker_error': type(exc).__name__, 'message': str(exc)}))
        sys.exit(1)
