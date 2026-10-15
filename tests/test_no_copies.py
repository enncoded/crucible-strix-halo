"""Ratchet against copied files (structure rule "No copies").

Every tracked file under tools/ and deploy/ that has a byte-identical twin there must be in
tests/data/copied_files.txt. The list may only get shorter: the test fails on a new copy (a path that is not listed)
and on a stale entry (a listed path that no longer has a twin, or no longer exists). To remove a copy, delete the
file or make it the single shared tool, then delete its line. Never add a line: extend the shared file instead.
"""
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ("tools", "deploy")
ALLOW = ROOT / "tests" / "data" / "copied_files.txt"


def tracked_files(root):
    """The tracked files of SCOPE. A git archive (cpu_checks.sh) has no .git: then every file on disk is tracked."""
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        top = None
    if top and Path(top).resolve() == root.resolve():
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", *SCOPE], capture_output=True,
                             check=True).stdout
        return sorted(name for name in out.decode().split("\0") if name)
    found = []
    for scope in SCOPE:
        for path in (root / scope).rglob("*"):
            if "__pycache__" not in path.parts and path.suffix != ".pyc":
                found.append(path.relative_to(root).as_posix())
    return sorted(found)


def copy_groups(root, names):
    """Groups (sorted tuples) of two or more paths with the same non-empty content. Symlinks do not count."""
    by_hash = {}
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        if data:
            by_hash.setdefault(hashlib.sha256(data).hexdigest(), []).append(name)
    return sorted(tuple(sorted(group)) for group in by_hash.values() if len(group) > 1)


def read_allow(path):
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def ratchet(groups, allowed):
    """(new, stale): copied paths that the list lacks, and listed paths that are no longer copies."""
    copied = {name for group in groups for name in group}
    return sorted(copied - set(allowed)), sorted(set(allowed) - copied)


class NoCopiesTest(unittest.TestCase):
    def test_no_new_copy_and_no_stale_entry(self):
        groups = copy_groups(ROOT, tracked_files(ROOT))
        allowed = read_allow(ALLOW)
        self.assertEqual(len(allowed), len(set(allowed)), "copied_files.txt has a duplicate line")
        new, stale = ratchet(groups, allowed)
        detail = "\n".join("  " + " = ".join(group) for group in groups if set(group) & set(new))
        self.assertEqual(new, [], "new byte-identical copies (extend the shared file instead):\n" + detail)
        self.assertEqual(stale, [], "stale entries in tests/data/copied_files.txt (delete these lines)")

    def test_ratchet_finds_new_and_stale_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, text in (("tools/a/x.py", "same"), ("tools/b/x.py", "same"), ("tools/c/y.py", "other"),
                               ("deploy/e.env", ""), ("deploy/f.env", "")):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text(text)
            (root / "tools/c/link.py").symlink_to(root / "tools/a/x.py")
            groups = copy_groups(root, tracked_files(root))
            self.assertEqual(groups, [("tools/a/x.py", "tools/b/x.py")])  # empty files and symlinks do not count
            self.assertEqual(ratchet(groups, ["tools/a/x.py", "tools/b/x.py"]), ([], []))
            self.assertEqual(ratchet(groups, ["tools/a/x.py"]), (["tools/b/x.py"], []))
            self.assertEqual(ratchet(groups, ["tools/a/x.py", "tools/b/x.py", "tools/c/y.py"]), ([], ["tools/c/y.py"]))


if __name__ == "__main__":
    unittest.main()
