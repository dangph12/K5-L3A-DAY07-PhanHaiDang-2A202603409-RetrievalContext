import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from costs import annotate, MODEL


class CostTests(unittest.TestCase):
    def test_price_missing_usage_and_idempotence(self):
        trace = dict(retriever='tfidf', model_calls=[
            dict(model=MODEL, usage=dict(prompt_token_count=19740, candidates_token_count=2888)),
            dict(model=MODEL, usage=None)])
        summary = dict(retriever='tfidf')
        annotate([trace], summary)
        self.assertEqual(summary['service_cost_usd'], 0.013142)
        self.assertNotIn('generation_pricing', summary)
        previous = repr((trace, summary))
        annotate([trace], summary)
        self.assertEqual(previous, repr((trace, summary)))

    def test_empty_calls_cost_zero(self):
        trace = dict(retriever='dense', model_calls=[])
        summary = dict(retriever='dense')
        annotate([trace], summary)
        self.assertEqual(summary['service_cost_usd'], 0.0)


if __name__ == '__main__':
    unittest.main()

