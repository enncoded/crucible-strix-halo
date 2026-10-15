"""CPU tests: a given build record must match the engine binary, for native engines too."""
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import gates as g

NATIVE_USAGE = '#!/bin/sh\necho "usage: engine --draft-head full|64k" >&2\nexit 2\n'


class BuildRecordShaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = self.root / 'engine'
        self.engine.write_text(NATIVE_USAGE)
        self.engine.chmod(self.engine.stat().st_mode | stat.S_IXUSR)

    def record(self, sha):
        path = self.root / 'record.json'
        path.write_text(json.dumps({'binary_sha256': sha}))
        return path

    def test_native_engine_with_matching_record_passes(self):
        policy = g.full_head_policy(self.engine, self.record(g.digest(self.engine)))
        self.assertTrue(policy['native'])

    def test_native_engine_with_mismatched_record_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'does not match the build record'):
            g.full_head_policy(self.engine, self.record('0' * 64))

    def test_native_engine_without_record_still_passes(self):
        self.assertTrue(g.full_head_policy(self.engine)['native'])


if __name__ == '__main__':
    unittest.main()
