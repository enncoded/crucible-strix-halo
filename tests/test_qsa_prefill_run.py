from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).parents[1]/'bench/qsa_prefill_engine_run.sh'
GATE_SCRIPT = Path(__file__).parents[1]/'bench/qsa_prefill_gate_run.sh'


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = self.root/'home'
        # The scripts take the queue wrapper from tools/work_paths.sh, which reads WORK_DIR.
        self.work = self.root/'work'
        self.work.joinpath('bin').mkdir(parents=True)
        self.home.mkdir()
        self.root.joinpath('results/prompts').mkdir(parents=True)
        for n in (16384, 262144):
            self.root.joinpath(f'results/prompts/long_{n}.txt').write_text('42\n')
        # This CPU fixture checks only the wrapper arguments. It does not load HIP.
        self.engine = self.root/'cpu_engine.py'
        self.engine.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
args=sys.argv[1:]
pathlib.Path(os.environ['HOME']+'/engine-args.json').write_text(json.dumps(args))
pathlib.Path(args[args.index('--dump-logits')+1]).write_bytes(bytes(4))
pathlib.Path(os.environ['CRUCIBLE_QSA_CAPTURE_DIR']+'/qsa_3.nsel.i32').write_bytes(bytes(4))
print('ids: 42')
''')
        self.engine.chmod(0o755)
        queue = self.work/'bin/gpu-run'
        queue.write_text('''#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys
args=sys.argv[1:]
pathlib.Path(os.environ['HOME']+'/queue-args.json').write_text(json.dumps(args))
subprocess.run(args[args.index('--')+1:], check=True)
''')
        queue.chmod(0o755)

    def run_script(self, tier, context, tag='case'):
        return subprocess.run(['bash', str(SCRIPT), str(self.engine), tier, context, tag], cwd=self.root,
                              env={**os.environ, 'HOME': str(self.home), 'WORK_DIR': str(self.work)}, text=True, capture_output=True)

    def test_invalid_tier_fails_before_queue(self):
        result = self.run_script('999', '16384')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('invalid tier', result.stderr)
        self.assertFalse(self.home.joinpath('queue-args.json').exists())

    def test_invalid_context_fails_before_queue(self):
        result = self.run_script('305', '262145')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('invalid context', result.stderr)
        self.assertFalse(self.home.joinpath('queue-args.json').exists())

    def test_invalid_tag_fails_before_queue(self):
        result = self.run_script('305', '16384', '../outside')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('invalid tag', result.stderr)
        self.assertFalse(self.home.joinpath('queue-args.json').exists())

    def test_gate_invalid_tag_fails_before_queue(self):
        result = subprocess.run(['bash', str(GATE_SCRIPT), str(self.engine), '/bin/false', '../outside', 'origin/main-test'],
                                cwd=self.root, env={**os.environ, 'HOME': str(self.home), 'WORK_DIR': str(self.work)},
                                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('invalid tag', result.stderr)
        self.assertFalse(self.home.joinpath('queue-args.json').exists())

    def test_regular_capture_priority_and_capacity(self):
        result = self.run_script('305', '16384')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.home.joinpath('engine-args.json').read_text())
        self.assertEqual(args[args.index('--ctx')+1], '16388')
        queue = json.loads(self.home.joinpath('queue-args.json').read_text())
        self.assertEqual(queue[:2], ['-p', '2'])
        self.assertIn('300', queue)

    def test_maximum_capture_uses_valid_capacity(self):
        result = self.run_script('405', '262144')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.home.joinpath('engine-args.json').read_text())
        self.assertEqual(args[args.index('--ctx')+1], '262144')
        queue = json.loads(self.home.joinpath('queue-args.json').read_text())
        self.assertEqual(queue[:2], ['-p', '3'])
        self.assertIn('600', queue)


if __name__ == '__main__':
    unittest.main()
