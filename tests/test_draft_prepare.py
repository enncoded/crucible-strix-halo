import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("prepare", Path(__file__).parents[1] / "tools/draft_prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class DraftVocabularyTest(unittest.TestCase):
    def test_blocks_preserve_hadamard_groups(self):
        self.assertEqual(prepare.rank_blocks({129: 10, 255: 9, 256: 18}, 128), [1])
        self.assertEqual(prepare.rank_blocks({129: 10, 255: 9, 256: 18}, 256), [1, 2])

    def test_ties_and_padding_are_deterministic(self):
        self.assertEqual(prepare.rank_blocks({129: 10, 256: 10}, 128), [1])
        self.assertEqual(prepare.rank_blocks({}, 256), [0, 1])

    def test_invalid_sizes_and_ids(self):
        for size in (0, -128, 129, 248448):
            with self.assertRaises(ValueError):
                prepare.rank_blocks({}, size)
        for counts in ({-1: 1}, {248320: 1}, {1: -1}):
            with self.assertRaises(ValueError):
                prepare.rank_blocks(counts, 128)


if __name__ == "__main__":
    unittest.main()
