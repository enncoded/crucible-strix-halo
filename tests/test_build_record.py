"""CPU tests: tools/build_record.py writes a build record that gates.py can check."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import build_record as br
import gates as g


class BuildRecordTests(unittest.TestCase):
    def test_record_matches_binary_and_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('engine/engine.hip', 'engine/model/flashnext/model.h', 'kernels/k.h', 'engine/exl3/safetensors.h', 'tools/x.py'):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text(name)
            binary = root / 'build/engine/crucible-x'
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'binary')
            with patch.object(br, 'compiler_lines', return_value=['HIP version: test']):
                path = br.write_record(binary, root, 'abc123', 'hipcc -o build/engine/crucible-x engine/engine.hip')
            record = json.loads(path.read_text())
        self.assertEqual(path.name, 'crucible-x.build.json')
        self.assertEqual(record['binary_sha256'], g.hashlib.sha256(b'binary').hexdigest())
        self.assertEqual(record['source_commit'], 'abc123')
        self.assertEqual(sorted(record['source_hashes']), sorted(['engine/exl3/safetensors.h', 'engine/engine.hip',
                                                                  'engine/model/flashnext/model.h', 'kernels/k.h']))
        self.assertEqual(record['compiler'], ['HIP version: test'])
        self.assertEqual(record['binary'], 'build/engine/crucible-x')

    def test_refuses_an_existing_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'engine').mkdir()
            binary = root / 'crucible-x'
            binary.write_bytes(b'b')
            (root / 'crucible-x.build.json').write_text('{}')
            with self.assertRaises(FileExistsError):
                br.write_record(binary, root, 'c', 'cmd')


if __name__ == '__main__':
    unittest.main()
