"""Run real Gemini-controlled retrieval; save traces for manual review."""
import argparse
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from agentic import run_agent
from baseline import TfidfIndex
from gemini_controller import GeminiController

ROOT = Path(__file__).resolve().parents[1]


class LexicalRetriever:
    def __init__(self, chunks):
        self.index = TfidfIndex(chunks)

    def search(self, query, *, filters=None, top_k=3):
        if filters:
            raise ValueError('Lexical agent currently supports unfiltered search only')
        return self.index.search(query, top_k=top_k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--question')
    source.add_argument('--cases', type=Path)
    parser.add_argument('--split', choices=['dev', 'test'])
    parser.add_argument('--retriever', choices=['tfidf', 'dense', 'hybrid'], default='dense')
    parser.add_argument('--model', default='gemini-3.5-flash-lite')
    parser.add_argument('--max-searches', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--budget', type=int, default=800)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.max_searches, args.top_k, args.budget) < 1:
        parser.error('Limits must be positive')
    if args.cases and not args.split:
        parser.error('--cases requires --split')
    load_dotenv(ROOT / '.env')
    if args.cases:
        cases = [c for c in json.loads(args.cases.read_text(encoding='utf-8-sig')) if c.get('split') == args.split]
    else:
        cases = [dict(id='interactive', question=args.question)]
    if not cases or len({c['id'] for c in cases}) != len(cases):
        parser.error('Cases must be nonempty with unique IDs')
    if any(not isinstance(c.get('question'), str) or not c['question'].strip() for c in cases):
        parser.error('Every case needs a nonempty question')
    args.output.mkdir(parents=True, exist_ok=True)
    traces = []
    retriever = None
    for case in cases:
        started = time.perf_counter()
        events = []
        controller = None
        try:
            if retriever is None:
                if args.retriever == 'tfidf':
                    chunks = json.loads((ROOT / 'data/day6/chunks.json').read_text(encoding='utf-8'))
                    retriever = LexicalRetriever(chunks)
                else:
                    from dense_retriever import DenseRetriever
                    from embeddings import MyEmbedder
                    dense = DenseRetriever(os.environ['DATABASE_URL'], MyEmbedder(), 'A')
                    chunks = dense.corpus()
                    if not chunks:
                        raise ValueError('No strategy A chunks')
                    if args.retriever == 'hybrid':
                        from hybrid_retriever import HybridRetriever
                        retriever = HybridRetriever(chunks, dense)
                    else:
                        retriever = dense
            controller = GeminiController(args.model, args.max_searches)
            # Only question text enters the agent: never gold, answerable or absence_reason.
            result = run_agent(case['question'], controller, retriever,
                max_searches=args.max_searches, top_k=args.top_k, budget=args.budget,
                event_log=events)
        except Exception as exc:
            result = dict(status='error', answer='', citations=[], events=events,
                          error_type=type(exc).__name__)
            # SDK error text can contain credentials/URLs; persist only its type/code.
            result['error_code'] = str(getattr(exc, 'code', ''))
        result.update(query_id=case['id'], question=case['question'], split=case.get('split'),
            retriever=args.retriever, model=args.model, database_strategy='A',
            max_searches=args.max_searches, top_k=args.top_k, budget_chars=args.budget,
            context_policy='cumulative, deduplicated, first-fit in retrieval order',
            model_calls=controller.calls if controller else [],
            elapsed_ms=(time.perf_counter()-started)*1000, service_cost_usd=None)
        traces.append(result)
        (args.output / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(dict(query_id=case['id'], status=result['status'], answer=result['answer'],
                             error_type=result.get('error_type')), ensure_ascii=False))
    summary = dict(model=args.model, retriever=args.retriever, cases=len(traces),
        answered=sum(t['status']=='answered' for t in traces),
        abstain=sum(t['status']=='abstain' for t in traces),
        errors=sum(t['status']=='error' for t in traces), service_cost_usd=None,
        note='Operational counts only. Human review required for grounding/citations/abstention correctness.')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if summary['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
