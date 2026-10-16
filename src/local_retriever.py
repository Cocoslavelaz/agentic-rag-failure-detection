"""Standalone adaptation tool; does not alter or invoke any MA-RAG graph."""
from dataclasses import asdict

from src.uncertainty import retrieval_features


def unexecuted_retrieval(reason='aggregate_branch'):
    return {'retrieval_occurred': False, 'retrieval_mask': False, 'missingness_reason': reason,
            'actual_returned_k': None, 'ordered_document_ids': None, 'raw_faiss_scores': None,
            'cosine_similarities': None, 'top1_uncertainty': None,
            'topk_mean_uncertainty': None, 'score_margin': None}


class LocalRetriever:
    def __init__(self, encoder, index, *, top_k=10):
        self.encoder, self.index, self.top_k = encoder, index, top_k
        self.last_search = None

    def search(self, query, *, top_k=None):
        k = self.top_k if top_k is None else top_k
        vectors = self.encoder.encode_queries([query])
        result = self.index.search(vectors, k=k)[0]
        validation = self.index.compare_cosine(vectors, k=k)
        scores = [hit['raw_score'] for hit in result['hits']]
        self.last_search = {
            'query': query, 'query_embedding_configuration': asdict(self.encoder.config),
            'retrieval_occurred': True, 'retrieval_mask': True, 'missingness_reason': None,
            'index_scope': self.index.manifest['scope'], 'corpus_checksum': self.index.manifest['corpus_checksum'],
            'top_k_requested': k, 'top_k_returned': result['actual_returned_k'],
            'actual_returned_k': result['actual_returned_k'],
            'ordered_document_ids': [hit['document_id'] for hit in result['hits']],
            'raw_faiss_scores': scores, 'cosine_similarities': scores,
            'faiss_device': 'cpu', 'document_norms_verified': True, 'query_norms_verified': True,
            'faiss_hits': result['hits'], 'missing_rows': result['missing_rows'],
            'cosine_validation': validation, **retrieval_features(scores, cosine_verified=validation['verified'])}
        return self.last_search

    def __call__(self, query):
        event = self.search(query)
        rows = [hit['row_id'] for hit in event['faiss_hits']]
        return [self.index.documents[row]['embedding_text'] for row in rows], event['ordered_document_ids']
