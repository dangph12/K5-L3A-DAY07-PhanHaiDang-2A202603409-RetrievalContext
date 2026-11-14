"""Offline ranking/contract checks; no mock benchmark quality is reported."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from hybrid_retriever import fuse, HybridRetriever


def chunk(key, text='apple'):
    return dict(id=key, text=text, source_id='doc', source='doc.pdf', page=1, metadata={}, score=999)


class HybridTests(unittest.TestCase):
    def test_agreement_promotes_chunk_and_ignores_raw_scores(self):
        result = fuse([chunk('a'), chunk('b')], [chunk('c'), chunk('b')])
        self.assertEqual(result[0]['id'], 'b')
        self.assertAlmostEqual(result[0]['score'], 2 / 62)
        self.assertEqual(result[0]['ranks'], {'tfidf': 2, 'dense': 2})
        self.assertEqual([c['id'] for c in result], ['b', 'a', 'c'])

    def test_no_double_count_within_one_method(self):
        result = fuse([chunk('a'), chunk('a')], [])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]['score'], 1 / 61)
        self.assertEqual(fuse([], []), [])

    def test_candidate_depth_and_output_limit(self):
        dense = Mock()
        dense.search.return_value = [chunk('b'), chunk('a')]
        retriever = HybridRetriever([chunk('a'), chunk('b', 'banana')], dense, candidate_k=10)
        result = retriever.search('apple', top_k=1)
        dense.search.assert_called_once_with('apple', top_k=10, filters=None)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['id'], 'a')

    def test_filters_and_backend_errors_are_explicit(self):
        dense = Mock()
        retriever = HybridRetriever([chunk('a')], dense)
        with self.assertRaises(ValueError):
            retriever.search('apple', filters={'chapter': 'X'})
        dense.search.assert_not_called()
        dense.search.side_effect = RuntimeError('connection failed')
        with self.assertRaisesRegex(RuntimeError, 'connection failed'):
            retriever.search('apple')


if __name__ == '__main__':
    unittest.main()
