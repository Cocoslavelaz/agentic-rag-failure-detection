import faiss 
import logging
import pickle
import glob
import os
import numpy as np
import hashlib
from src.telemetry import ACTIVE_TRACE, digest

from accelerate.logging import get_logger

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self, 
        gpu_ids
    ):
        logger.info("Initializing retriever")
        self.doc_lookup = []
        self.gpu_ids = gpu_ids
        self.document_norms_verified = None
        self.index_version = None
        self.last_search = None

    def _initialize_faiss_index(self, dim: int):
        self.index = None
        cpu_index = faiss.IndexFlatIP(dim)
        self.index = cpu_index

    def _move_index_to_gpu(self):
        ngpus = len(self.gpu_ids)
        gpu_ids = self.gpu_ids
        total = faiss.get_num_gpus()
        logger.info(f"Moving index to GPU(s) {ngpus} {total}")
        # print("total gpu", total)
        gpu_resources = []
        assert ngpus <= total, "Number of gpus for retrieval must less than total gpus"
        for i in range(ngpus):
            res = faiss.StandardGpuResources()
            gpu_resources.append(res)
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpus):
            vdev.push_back(i)
            vres.push_back(gpu_resources[gpu_ids[i]])
        self.index = faiss.index_cpu_to_gpu_multiple(vres, vdev, self.index, co)
        self.index_on_gpu = True
    
    def init_index_and_add(self, root_dir, dataset_name):
        logger.info("Initializing Faiss index from pre-computed document embeddings")
        partitions = glob.glob(os.path.join(root_dir, f"{dataset_name}*"))
        audit = ACTIVE_TRACE.get() is not None or os.getenv("TRAJECTORY_ENABLED", "false").lower() == "true"
        verified = bool(partitions)
        fingerprints = []
        for i, part in enumerate(partitions):
            logger.info(f"Load {part}")
            with open(part, "rb") as f:
                data = pickle.load(f)
            lookup_indices = data[0]
            encoded = data[1]
            if audit:
                norms = np.linalg.norm(encoded, axis=1)
                verified = verified and bool(len(norms)) and bool(np.all(np.isfinite(norms)) and np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5))
                with open(part, "rb") as handle:
                    fingerprints.append({"partition": os.path.basename(part), "sha256": hashlib.file_digest(handle, "sha256").hexdigest()})
            # print(encoded.shape)
            # print(lookup_indices.shape)
            if i == 0:
                dim = encoded.shape[1]
                self._initialize_faiss_index(dim)
            self.index.add(encoded)
            self.doc_lookup.extend(lookup_indices)
            # break
        if audit:
            self.document_norms_verified = verified
            self.index_version = digest(fingerprints)
        self._move_index_to_gpu()
    
    def search(self, query_ids, query_embs, top_k):
        return_dict = {}
        D, I = self.index.search(query_embs, top_k)
        original_indices = np.array(self.doc_lookup)[I]
        # Sidecar only: preserve the original search result and dictionary order.
        query_norms = np.linalg.norm(query_embs, axis=1)
        self.last_search = {"index_type": type(self.index).__name__, "score_metric": "inner_product",
                            "index_version": self.index_version,
                            "document_norms_verified": self.document_norms_verified,
                            "query_norms_verified": bool(np.all(np.isfinite(query_norms)) and np.allclose(query_norms, 1.0, rtol=1e-5, atol=1e-5)),
                            "faiss_hits": [[{"index_position": int(pos), "document_id": str(doc), "score": float(score),
                                             "valid_index_position": bool(pos >= 0)}
                                            for pos, doc, score in zip(positions, docs, scores)]
                                           for positions, docs, scores in zip(I, original_indices, D)]}
        for qid, scores_per_q, doc_indices_per_q in zip(query_ids, D, original_indices):
            return_dict[qid] = {}
            for doc_index, score in zip(doc_indices_per_q, scores_per_q):
                doc_index = str(doc_index)
                return_dict[qid][doc_index] = float(score)

        return return_dict
    

if __name__ == "__main__":
    retrieve = Retriever()
    os.makedirs("save_embs", exist_ok=True)
    retrieve.init_index_and_add(root_dir="save_embs/gte-ml-base", dataset_name="dpr100")
