"""Offline checks of explicit abstention and partial failure traces."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from agentic import run_agent


class RuntimeTests(unittest.TestCase):
    def test_explicit_abstention_retains_search(self):
        controller = Mock()
        controller.decide.side_effect = [dict(action='search', query='q'),
                                        dict(action='abstain', answer='missing evidence')]
        retriever = Mock()
        retriever.search.return_value = []
        result = run_agent('q', controller, retriever)
        self.assertEqual(result['status'], 'abstain')
        self.assertEqual(result['citations'], [])
        self.assertEqual(result['events'][0]['event'], 'search')
        self.assertEqual(result['events'][-1]['reason'], 'missing evidence')

    def test_error_keeps_partial_events(self):
        controller = Mock()
        controller.decide.return_value = dict(action='search', query='q')
        retriever = Mock()
        retriever.search.side_effect = ConnectionError('unavailable')
        events = []
        with self.assertRaises(ConnectionError):
            run_agent('q', controller, retriever, event_log=events)
        self.assertEqual(events[0]['event'], 'search_error')
        self.assertEqual(events[0]['query'], 'q')


if __name__ == '__main__':
    unittest.main()
