import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from signer.metrics import score


def row(end=6, entities=True):
    return dict(id='a', language='uz', text='Londonning',
                entities=[dict(start=0, end=end, label='LOC')] if entities else [])


class MetricsTests(unittest.TestCase):
    def test_partial_overlap_is_not_match(self):
        result = score([row(end=10)], [row(end=6)])
        self.assertEqual(result['overall']['micro_f1'], 0.)
        self.assertEqual(result['overall']['reference'], 1)

    def test_missing_prediction_rejected(self):
        with self.assertRaises(ValueError):
            score([], [row()])

    def test_empty_gold_false_positive_counted(self):
        result = score([row()], [row(entities=False)])
        self.assertEqual(result['overall']['predicted'], 1)
        self.assertEqual(result['overall']['matched'], 0)

    def test_perfect_match(self):
        self.assertEqual(score([row()], [row()])['overall']['micro_f1'], 1.)
