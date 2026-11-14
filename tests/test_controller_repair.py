import sys
import unittest
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from gemini_controller import GeminiController


class RepairTests(unittest.TestCase):
    def controller(self, actions):
        c = GeminiController.__new__(GeminiController)
        c.max_searches = 4
        c.calls = [{}]
        c._request = Mock(side_effect=actions)
        return c

    def test_missing_inline_citation_is_repaired(self):
        c = self.controller([dict(action='answer', answer='Seven', citations=['c1']),
                             dict(action='answer', answer='Seven [c1]', citations=['c1'])])
        action = c.decide('q', [dict(queries=['q'], context_ids=['c1'], context='Seven')])
        self.assertEqual(action['answer'], 'Seven [c1]')
        self.assertEqual(c._request.call_count, 2)
        self.assertIn('validation_feedback', c._request.call_args.args[0])

    def test_invented_citations_never_accepted(self):
        bad = dict(action='answer', answer='Seven [fake]', citations=['fake'])
        c = self.controller([bad, bad])
        with self.assertRaises(ValueError):
            c.decide('q', [dict(queries=['q'], context_ids=['c1'], context='Seven')])

    def test_first_query_preserves_original(self):
        c = self.controller([dict(action='search', query='rewrite'), dict(action='search', query='original')])
        self.assertEqual(c.decide('original', [])['query'], 'original')


if __name__ == '__main__':
    unittest.main()
