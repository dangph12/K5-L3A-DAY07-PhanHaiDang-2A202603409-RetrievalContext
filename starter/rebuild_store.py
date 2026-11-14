"""Rebuild strategy A on a fresh pgvector database from permitted shared chunks."""
import json
import os
from pathlib import Path
import psycopg
from psycopg.types.json import Jsonb
from dotenv import load_dotenv
from embeddings import MyEmbedder

ROOT = Path(__file__).resolve().parents[1]


def main():
    load_dotenv(ROOT / '.env')
    chunks = json.loads((ROOT/'data/day6/chunks.json').read_text(encoding='utf-8'))
    with psycopg.connect(os.environ['DATABASE_URL']) as conn:
        conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        conn.execute('''CREATE TABLE IF NOT EXISTS day6_chunks (
            strategy text NOT NULL, chunk_id text NOT NULL, document_id text NOT NULL,
            source text NOT NULL, page integer NOT NULL, text text NOT NULL,
            metadata jsonb NOT NULL, embedding vector(768) NOT NULL,
            PRIMARY KEY(strategy, chunk_id))''')
        if conn.execute("SELECT count(*) FROM day6_chunks WHERE strategy='A'").fetchone()[0]:
            raise ValueError('Strategy A exists; refusing to overwrite. Use prepare_store.py instead.')
        vectors = MyEmbedder().embed_documents([c['text'] for c in chunks])
        for c, v in zip(chunks, vectors):
            conn.execute('INSERT INTO day6_chunks VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector)',
                ('A', c['id'], c['source_id'], c['source'], c['page'], c['text'], Jsonb(c['metadata']), json.dumps(v)))
    print(f'Rebuilt strategy A: {len(chunks)} chunks')


if __name__ == '__main__':
    main()
