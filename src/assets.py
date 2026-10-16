"""Local-only readiness checks, with no model/data acquisition or rebuilding."""
from pathlib import Path

DPR_CORPUS_ID = "dpr-w100/natural-questions/dev"
ORIGINAL_EMBEDDING_PATH = "hf/gte-multilingual-base"
STORE_FILES = ("bin", "bin.pos", "idx.doc_id", "bin.meta")


def corpus_store_path(path):
    if not path:
        raise FileNotFoundError("Set --corpus-store / DPR_CORPUS_STORE to the existing DPR .pklz4 docstore directory")
    path = Path(path).expanduser().resolve()
    missing = [name for name in STORE_FILES if not (path/name).is_file()]
    if missing:
        raise FileNotFoundError(f"DPR docstore incomplete at {path}: missing {missing}; no download or build")
    return path


def open_dpr_corpus(path):
    path = corpus_store_path(path)
    from ir_datasets.indices import PickleLz4FullStore
    from ir_datasets.datasets.dpr_w100 import DprW100Doc
    def no_build():
        raise RuntimeError("Automatic DPR docstore rebuilding is disabled")
    store = PickleLz4FullStore(str(path), init_iter_fn=no_build, data_cls=DprW100Doc,
                              lookup_field="doc_id", index_fields=["doc_id"])
    if not store.built():
        raise ValueError(f"Empty DPR docstore: {path}; no automatic build")
    return store


def local_embedding_path(path):
    path = Path(path).expanduser().resolve()
    if not path.is_dir() or not (path/"config.json").is_file():
        raise FileNotFoundError(f"Original embedding model directory not found: {path}; no model download/substitution")
    if not any(path.glob("*.safetensors")) and not any(path.glob("pytorch_model*.bin")):
        raise FileNotFoundError(f"No local embedding weights at {path}")
    return path
