"""Ratchet against near-copies (structure rule "No copies").

tests/test_no_copies.py finds byte-identical twins. A copy that changes one line (a train number, a path) passes that
test. This test compares the text files of tools/ and deploy/ line by line: two files of the same suffix with a
difflib ratio of THRESHOLD or more are a near-copy pair. Every pair must be in tests/data/near_copies.txt. The list
may only get shorter: the test fails on a new pair and on a stale line (a pair that is gone or has changed enough).
To remove a pair, delete a file or make it the single shared tool, then delete its line. Never add a line: extend the
shared file instead. Byte-identical pairs belong to test_no_copies.py and do not count here.
"""
import difflib
import itertools
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_no_copies import ROOT, read_allow, tracked_files  # noqa: E402

ALLOW = ROOT / "tests" / "data" / "near_copies.txt"
THRESHOLD = 0.9
SKIP_SUFFIXES = {".json"}  # generated data (every train adds a paths_baseline_tN.json, a test keeps its input next to it)
MAX_BYTES = 400_000  # a larger file is data, and the quadratic match would be slow


def read_lines(root, names):
    """name -> lines for each tracked text file that is not empty, not a symlink, not too big and not generated data
    (SKIP_SUFFIXES)."""
    lines = {}
    for name in names:
        path = root / name
        if path.suffix in SKIP_SUFFIXES or path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if text:
            lines[name] = text.splitlines()
    return lines


def near_pairs(lines, threshold=THRESHOLD):
    """Sorted "A B" strings (A < B) for the files of one suffix with a line ratio of threshold or more. A pair with
    equal lines is skipped: test_no_copies.py owns it."""
    by_suffix = {}
    for name in lines:
        by_suffix.setdefault(Path(name).suffix, []).append(name)
    found = []
    for names in by_suffix.values():
        for a, b in itertools.combinations(sorted(names), 2):
            if lines[a] == lines[b]:
                continue
            match = difflib.SequenceMatcher(None, lines[a], lines[b], autojunk=False)
            if match.real_quick_ratio() >= threshold and match.quick_ratio() >= threshold \
                    and match.ratio() >= threshold:
                found.append(f"{a} {b}")
    return sorted(found)


def ratchet(found, allowed):
    """(new, stale): pairs that the list lacks, and listed pairs that are no longer near-copies."""
    return sorted(set(found) - set(allowed)), sorted(set(allowed) - set(found))


class NearCopiesTest(unittest.TestCase):
    def test_no_new_near_copy_and_no_stale_entry(self):
        found = near_pairs(read_lines(ROOT, tracked_files(ROOT)))
        allowed = read_allow(ALLOW)
        self.assertEqual(len(allowed), len(set(allowed)), "near_copies.txt has a duplicate line")
        new, stale = ratchet(found, allowed)
        self.assertEqual(new, [], "new near-copy pairs (extend the shared file instead):\n  " + "\n  ".join(new))
        self.assertEqual(stale, [], "stale lines in tests/data/near_copies.txt (delete these lines)")

    def test_pairs_are_sorted_and_use_two_tracked_paths(self):
        tracked = set(tracked_files(ROOT))
        for line in read_allow(ALLOW):
            first, second = line.split(" ")
            self.assertLess(first, second, line)
            self.assertTrue({first, second} <= tracked, line)

    def test_ratchet_finds_near_pairs_and_skips_other_files(self):
        base = [f"line {i}" for i in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {"tools/a/paths_baseline_t1.json": base, "tools/b/paths_baseline_t2.json": base[:5] + ["x"] + base[6:],
                     "tools/a/x.py": base, "tools/b/x.py": base[:5] + ["changed"] + base[6:],  # 1 of 20 lines
                     "tools/c/x.py": base[:10] + ["other"] * 10,  # half the lines
                     "tools/d/x.py": base, "tools/e/x.sh": base[:5] + ["changed"] + base[6:], "deploy/f.bin": []}
            for name, content in files.items():
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text("\n".join(content) + ("\n" if content else ""))
            (root / "tools/g.py").symlink_to(root / "tools/a/x.py")
            found = near_pairs(read_lines(root, tracked_files(root)))
            # a = d (identical, skipped); b is near a and d; c is far; the .sh file has another suffix; the two
            # .json files are generated data and do not count
            self.assertEqual(found, ["tools/a/x.py tools/b/x.py", "tools/b/x.py tools/d/x.py"])
            self.assertEqual(ratchet(found, found), ([], []))
            self.assertEqual(ratchet(found, found[:1]), (found[1:], []))
            self.assertEqual(ratchet(found, found + ["tools/c/x.py tools/d/x.py"]), ([], ["tools/c/x.py tools/d/x.py"]))


if __name__ == "__main__":
    unittest.main()
