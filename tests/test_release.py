import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_release import verify_no_secrets, verify_sample, verify_public_data


class ReleaseTests(unittest.TestCase):
    def test_sample(self):
        self.assertEqual(verify_sample(), [])

    def test_public_examples_unchanged(self):
        self.assertEqual(verify_public_data(), [])

    def test_no_secrets(self):
        self.assertEqual(verify_no_secrets(), [])
