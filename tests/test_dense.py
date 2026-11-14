"""Mock-only checks; these do not measure dense retrieval quality."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from dense_retriever import DenseRetriever
from baseline import evaluate


class DenseTests(unittest.TestCase):
    def test_query_vector_filter_and_schema(self):
        embedder = Mock()
        embedder.embed_query.return_value = [1.0] * 768
        with patch('dense_retriever.psycopg.connect') as connect:
            conn = connect.return_value.__enter__.return_value
            conn.execute.return_value.fetchall.return_value = [
                ('c1', 'doc', 'doc.pdf', 1, 'Evidence here.', {}, 0.9)]
            index = DenseRetriever('unused', embedder)
            result = index.search('question', top_k=2, filters={'chapter': 'X'})
            self.assertEqual(result[0]['id'], 'c1')
            self.assertEqual(result[0]['source_id'], 'doc')
            params = conn.execute.call_args.args[1]
            self.assertEqual(params[1], 'A')
            self.assertEqual(params[2].obj, {'chapter': 'X'})
            self.assertEqual(params[-1], 2)
            self.assertTrue(conn.read_only)
            embedder.embed_query.assert_called_once_with('question')

    def test_invalid_vector_fails_before_database(self):
        embedder = Mock()
        embedder.embed_query.return_value = [0.0] * 768
        with patch('dense_retriever.psycopg.connect') as connect:
            with self.assertRaises(ValueError):
                DenseRetriever('unused', embedder).search('question')
            connect.assert_not_called()

    def test_evaluation_sends_only_question(self):
        index = Mock()
        index.search.return_value = []
        q = dict(id='q', split='dev', question='public question', gold=[
            dict(source_id='doc', page=1, evidence='secret gold')])
        evaluate(index, [q])
        index.search.assert_called_once_with('public question', top_k=3)


if __name__ == '__main__':
    unittest.main()
