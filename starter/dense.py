"""Evaluate dense retrieval with the same top-3/context policy as TF-IDF."""
import argparse
import json
import os
import statistics
from pathlib import Path
from dotenv import load_dotenv
from baseline import evaluate, validate
from dense_retriever import DenseRetriever
from embeddings import MyEmbedder

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, default=ROOT / 'data/day6/questions-baseline.json')
    parser.add_argument('--split', required=True, choices=['dev', 'test'])
    parser.add_argument('--strategy', default='A')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / '.env')
    questions = json.loads(args.questions.read_text(encoding='utf-8'))
    selected = [q for q in questions if q.get('split') == args.split]
    retriever = DenseRetriever(os.environ['DATABASE_URL'], MyEmbedder(), args.strategy)
    # Validate against the actual database corpus, not a potentially stale JSON export.
    validate(retriever.corpus(), selected)
    # evaluate sends only question text to search; no gold or benchmark filters.
    traces = evaluate(retriever, selected)
    for trace in traces:
        trace['strategy'] = 'dense'
    summary = dict(strategy='dense', split=args.split, n=len(traces),
        hit1=sum(t['hit1'] for t in traces), hit3=sum(t['hit3'] for t in traces),
        context_hit=sum(t['context_hit'] for t in traces),
        avg_context_chars=statistics.mean(t['context_chars'] for t in traces),
        median_ms=statistics.median(t['elapsed_ms'] for t in traces),
        index_seconds=None, service_cost_usd=None,
        embedding_model=MyEmbedder.model_name, dimensions=768,
        database_strategy=args.strategy, filters_applied=False,
        timing_note='Includes query embedding API, database connection/search and context packing; existing document index reused.',
        cost_note='Gemini API cost not measured; consult provider billing.')
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [('summary.json', summary), ('traces.json', traces)]:
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
