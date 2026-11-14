import json
import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values

day6 = Path(os.environ["DAY6_DIR"])
day7 = Path(os.environ["DAY7_DIR"])
output = day7 / "starter/data/day6_export"
output.mkdir(parents=True, exist_ok=True)

config = dotenv_values(day6 / "starter/.env")
database_url = os.environ.get("DATABASE_URL") or config.get("DATABASE_URL")
if not database_url:
    raise RuntimeError("Thieu DATABASE_URL trong starter/.env cua Day06")

with psycopg.connect(database_url, row_factory=dict_row) as conn:
    chunks = conn.execute("""
        SELECT chunk_id, document_id, source, page, text, metadata
        FROM day6_chunks
        WHERE strategy = %s
        ORDER BY chunk_id
    """, ("A",)).fetchall()

if not chunks:
    raise RuntimeError("Database khong co chunks strategy A")

raw = json.loads(
    (day6 / "benchmark/questions.json").read_text(encoding="utf-8")
)
questions = [item["question"] for item in raw]

for name, data in (
    ("chunks.json", chunks),
    ("questions.json", questions),
):
    (output / name).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

print(f"Da xuat {len(chunks)} chunks va {len(questions)} cau hoi")
print(output)
