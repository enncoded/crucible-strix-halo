import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("startup_runs", ROOT / "bench/explore/startup_runs.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class InsideTests(unittest.TestCase):
    def test_fresh_processes_and_complete_artifacts(self):
        phases = ['engine_ready', 'resident_ready', 'model_load', 'plan_build']
        stdout = '\n'.join(json.dumps({'phase': x, 'seconds': 1.5}) for x in phases) + '\n'
        done = mock.Mock(returncode=0, stdout=stdout, stderr='ready\n')
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            with mock.patch.object(runner, 'system_state', return_value={'loadavg': [0, 0, 0]}), \
                    mock.patch.object(runner.subprocess, 'run', return_value=done) as execute, \
                    mock.patch.dict(runner.os.environ, {'HSA_USE_SVM': '0', 'CRUCIBLE_PROFILE': '1'}):
                runner.inside(Path('/engine'), Path('/checkpoint'), [0, 4], target)
            self.assertEqual(execute.call_count, 2)
            self.assertEqual(execute.call_args_list[0].args[0], ['/engine', '/checkpoint', '0'])
            self.assertEqual(execute.call_args_list[1].args[0], ['/engine', '/checkpoint', '4'])
            self.assertNotIn('HSA_USE_SVM', execute.call_args.kwargs['env'])
            self.assertNotIn('CRUCIBLE_PROFILE', execute.call_args.kwargs['env'])
            for prefix in ('0-0', '1-4'):
                self.assertEqual((target / f'{prefix}.stdout').read_text(), stdout)
                self.assertEqual((target / f'{prefix}.stderr').read_text(), 'ready\n')
                self.assertEqual(json.loads((target / f'{prefix}.system.json').read_text())['exit'], 0)

    def test_failed_child_keeps_artifacts(self):
        done = mock.Mock(returncode=124, stdout='partial\n', stderr='timeout\n')
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            with mock.patch.object(runner, 'system_state', return_value={}), \
                    mock.patch.object(runner.subprocess, 'run', return_value=done):
                with self.assertRaisesRegex(RuntimeError, 'exit 124'):
                    runner.inside(Path('/engine'), Path('/checkpoint'), [0], target)
            self.assertEqual((target / '0-0.stderr').read_text(), 'timeout\n')
            self.assertTrue((target / '0-0.command.json').is_file())

    def test_missing_phases_fail_closed(self):
        done = mock.Mock(returncode=0, stdout='{}\n', stderr='')
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(runner, 'system_state', return_value={}), \
                    mock.patch.object(runner.subprocess, 'run', return_value=done):
                with self.assertRaisesRegex(RuntimeError, 'missing phases'):
                    runner.inside(Path('/engine'), Path('/checkpoint'), [0], Path(directory))


if __name__ == '__main__':
    unittest.main()
