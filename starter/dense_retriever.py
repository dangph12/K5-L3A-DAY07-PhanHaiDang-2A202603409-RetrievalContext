"""Read-only dense retrieval over the existing Day06 pgvector corpus."""
import math
import psycopg
from psycopg.types.json import Jsonb


class DenseRetriever:
    def __init__(self, database_url, embedder, strategy='A'):
        self.database_url = database_url
        self.embedder = embedder
        self.strategy = strategy

    def corpus(self):
        with psycopg.connect(self.database_url) as conn:
            conn.read_only = True
            dimension = conn.execute("SELECT atttypmod FROM pg_attribute WHERE attrelid='day6_chunks'::regclass AND attname='embedding'").fetchone()
            if dimension is None or dimension[0] != 768:
                raise ValueError('Expected vector(768) in day6_chunks')
            rows = conn.execute('SELECT chunk_id, document_id, source, page, text, metadata FROM day6_chunks WHERE strategy=%s ORDER BY chunk_id', (self.strategy,)).fetchall()
        return [dict(id=r[0], source_id=r[1], source=r[2], page=r[3], text=r[4], metadata=r[5]) for r in rows]

    def search(self, query, *, top_k=3, filters=None):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Query must be nonempty')
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k must be a positive integer')
        values = self.embedder.embed_query(query)
        if len(values) != 768 or not all(math.isfinite(v) for v in values) or not any(values):
            raise ValueError('Query embedding must have 768 finite, nonzero dimensions')
        vector = '[' + ','.join(str(float(v)) for v in values) + ']'
        with psycopg.connect(self.database_url) as conn:
            conn.read_only = True
            rows = conn.execute('''SELECT chunk_id, document_id, source, page, text, metadata,
                1 - (embedding <=> %s::vector)
                FROM day6_chunks WHERE strategy=%s AND metadata @> %s
                ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s''',
                (vector, self.strategy, Jsonb(filters or {}), vector, top_k)).fetchall()
        return [dict(id=r[0], source_id=r[1], source=r[2], page=r[3], text=r[4], metadata=r[5],
                     score=float(r[6]), strategy='dense') for r in rows]
