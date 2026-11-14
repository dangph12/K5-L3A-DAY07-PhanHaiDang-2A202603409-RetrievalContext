"""Evaluate hybrid RRF with the baseline top-3 and 800-character context policy."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path
from dotenv import load_dotenv
from baseline import evaluate, validate
from dense_retriever import DenseRetriever
from embeddings import MyEmbedder
from hybrid_retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, default=ROOT / 'data/day6/questions-baseline.json')
    parser.add_argument('--split', required=True, choices=['dev', 'test'])
    parser.add_argument('--strategy', default='A')
    parser.add_argument('--candidate-k', type=int, default=10)
    parser.add_argument('--rrf-k', type=int, default=60)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.candidate_k < 1 or args.rrf_k < 1:
        parser.error('candidate-k and rrf-k must be positive')
    load_dotenv(ROOT / '.env')
    selected = [q for q in json.loads(args.questions.read_text(encoding='utf-8')) if q.get('split') == args.split]
    dense = DenseRetriever(os.environ['DATABASE_URL'], MyEmbedder(), args.strategy)
    chunks = dense.corpus()
    validate(chunks, selected)
    start = time.perf_counter()
    retriever = HybridRetriever(chunks, dense, candidate_k=args.candidate_k, rrf_k=args.rrf_k)
    index_seconds = time.perf_counter() - start
    traces = evaluate(retriever, selected)
    for trace in traces:
        trace.update(strategy='hybrid', candidate_k=args.candidate_k, rrf_k=args.rrf_k)
    summary = dict(strategy='hybrid', split=args.split, n=len(traces),
        hit1=sum(t['hit1'] for t in traces), hit3=sum(t['hit3'] for t in traces),
        context_hit=sum(t['context_hit'] for t in traces),
        avg_context_chars=statistics.mean(t['context_chars'] for t in traces),
        median_ms=statistics.median(t['elapsed_ms'] for t in traces),
        index_seconds=index_seconds, service_cost_usd=None,
        embedding_model=MyEmbedder.model_name, dimensions=768,
        database_strategy=args.strategy, filters_applied=False,
        candidate_k=args.candidate_k, rrf_k=args.rrf_k,
        timing_note='Index time is lexical index creation only; retrieval includes Gemini API, lexical search, database connection/search, fusion and context packing.',
        cost_note='Gemini API cost not measured; consult provider billing.')
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [('summary.json', summary), ('traces.json', traces)]:
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
