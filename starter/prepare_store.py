"""Read-only check of the existing Day06 pgvector store."""

import argparse
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="A")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        conn.read_only = True
        dimension = conn.execute(
            "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'day6_chunks'::regclass AND attname = 'embedding'"
        ).fetchone()
        if dimension is None or dimension[0] != 768:
            raise ValueError("Expected day6_chunks.embedding vector(768)")
        count = conn.execute(
            "SELECT count(*) FROM day6_chunks WHERE strategy = %s", (args.strategy,)
        ).fetchone()[0]
        if not count:
            raise ValueError("Selected strategy has no chunks")
    print(
        f"Connected: day6_chunks, strategy={args.strategy}, chunks={count}, dimensions=768"
    )


if __name__ == "__main__":
    main()
