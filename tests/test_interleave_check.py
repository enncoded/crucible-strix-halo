import tempfile
import unittest
from pathlib import Path

from tools import interleave_check


class InterleaveCheckTest(unittest.TestCase):
    def files(self, a, b):
        d = Path(tempfile.mkdtemp())
        (d / "a").write_text(a)
        (d / "b").write_text(b)
        return ["x", str(d / "a"), str(d / "b")]

    def test_equal_keys_pass(self):
        a = "row 01 10 5 aa\nchunk 02 0 4096 bb cc\nrow 01 11 6 dd\n"
        b = "row 01 10 5 aa\nchunk 02 0 4096 bb cc\nrow 03 7 1 ee\n"
        self.assertEqual(interleave_check.main(self.files(a, b)), 0)

    def test_mismatch_fails(self):
        self.assertEqual(interleave_check.main(self.files("row 01 10 5 aa\n", "row 01 10 5 ab\n")), 1)
        self.assertEqual(interleave_check.main(self.files("chunk 02 0 9 bb cc\n", "chunk 02 0 9 bb cd\n")), 1)

    def test_conflict_within_a_file_fails(self):
        self.assertEqual(interleave_check.main(self.files("row 01 10 5 aa\nrow 01 10 5 ab\n", "row 01 10 5 aa\n")), 1)

    def test_no_shared_key_fails(self):
        self.assertEqual(interleave_check.main(self.files("row 01 10 5 aa\n", "row 01 11 5 aa\n")), 1)

    def test_malformed_line_raises(self):
        with self.assertRaises(ValueError):
            interleave_check.main(self.files("row 01\n", "row 01 10 5 aa\n"))


if __name__ == "__main__":
    unittest.main()
