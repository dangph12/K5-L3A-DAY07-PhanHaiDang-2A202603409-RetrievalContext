"""Equal-weight reciprocal rank fusion of lexical and dense candidates."""
from baseline import TfidfIndex


def fuse(lexical, dense, *, top_k=3, rrf_k=60):
    if type(top_k) is not int or top_k < 1 or type(rrf_k) is not int or rrf_k < 1:
        raise ValueError('top_k and rrf_k must be positive integers')
    merged = {}
    for name, results in [('tfidf', lexical), ('dense', dense)]:
        seen = set()
        for rank, chunk in enumerate(results, 1):
            key = chunk['id']
            if key in seen:
                continue
            seen.add(key)
            if key not in merged:
                merged[key] = {**chunk, 'score': 0.0, 'strategy': 'hybrid', 'ranks': {}}
            merged[key]['score'] += 1 / (rrf_k + rank)
            merged[key]['ranks'][name] = rank
    return sorted(merged.values(), key=lambda c: (-c['score'], c['id']))[:top_k]


class HybridRetriever:
    def __init__(self, chunks, dense, *, candidate_k=10, rrf_k=60):
        if type(candidate_k) is not int or candidate_k < 1 or type(rrf_k) is not int or rrf_k < 1:
            raise ValueError('candidate_k and rrf_k must be positive integers')
        self.lexical = TfidfIndex(chunks)
        self.dense = dense
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(self, query, *, top_k=3, filters=None):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Query must be nonempty')
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k must be a positive integer')
        if filters:
            raise ValueError('Hybrid currently supports unfiltered retrieval only')
        count = max(top_k, self.candidate_k)
        lexical = self.lexical.search(query, top_k=count)
        dense = self.dense.search(query, top_k=count, filters=None)
        return fuse(lexical, dense, top_k=top_k, rrf_k=self.rrf_k)
