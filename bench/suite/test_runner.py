import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("suite_run", HERE / "run.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


# CRUCIBLE_PI_AI_DIR and CRUCIBLE_TEST_CKPT only enable optional server tests in the full `make test-cpu`
# environment. The runner refuses unknown CRUCIBLE_* names, so clear them here instead of widening the allow-list.
_TEST_DEPENDENCY_ENV = patch.dict(os.environ)


def setUpModule():
    _TEST_DEPENDENCY_ENV.start()
    for name in ("CRUCIBLE_PI_AI_DIR", "CRUCIBLE_TEST_CKPT"):
        os.environ.pop(name, None)


def tearDownModule():
    _TEST_DEPENDENCY_ENV.stop()


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        (self.checkpoint / "config.json").write_text("{}")
        (self.checkpoint / "quantization_config.json").write_text("{}")
        self.binary = self.root / "resident"
        self.binary.write_bytes(b"test executable")
        suite.save(Path(str(self.binary) + ".build.json"), {
            "binary_sha256": suite.digest(self.binary), "source_branch": "test", "source_commit": "abc"})
        self.args = argparse.Namespace(checkpoint=self.checkpoint, tokens=3, chunk=8, spec=2,
                                       repeat=1, warmup=0, serial_ceiling=50,
                                       prefill_ceiling=100, spec_ceiling=100, gpu_run=Path("/fake/gpu-run"))
        self.corpus = suite.load_corpus(HERE / "manifest.json")
        self.prompts = self.corpus["prompts"][:1]

    def fake_run(self, command, stdout, stderr):
        self.assertEqual(command[:10], ["/fake/gpu-run", "-p", str(getattr(self.args, 'priority', 3)), "-w", "3600", "--", "timeout", "-k", "10", "300"])
        self.assertIn("_worker", command)
        suite.save(Path(command[-2]), {"host": "fake", 'environment_policy_hashes':suite.policy_hashes()})
        stdout.write(json.dumps({"event": "ready", "load_seconds": 1}) + "\n")
        for spec in (0, 2):
            row = {"event": "result", "name": self.prompts[0]["name"], "repeat": 0, "spec": spec,
                   "prompt_tokens": self.prompts[0]["tokens"], "output_ids": [1, 2, 3],
                   "prefill_seconds": 1, "decode_seconds": 1, "ttft_seconds": 1,
                   "accepted": 1 if spec else 0, "proposed": 2 if spec else 0, "forwards": 2}
            stdout.write(json.dumps(row) + "\n")
        return subprocess.CompletedProcess(command, 0)

    def test_end_to_end_report_and_comparison_cpu_only(self):
        with patch.object(suite.subprocess, "run", side_effect=self.fake_run):
            report = suite.run_binary(self.binary, self.root / "a", self.prompts, self.corpus, self.args)
        self.assertEqual(len(report["rows"]), 2)
        self.assertTrue((self.root / "a/results.md").exists())
        suite.save_comparison(self.root / "delta.json", report, report)
        self.assertEqual(json.loads((self.root / "delta.json").read_text())[0]["decode_change_pct"], 0)
        self.assertIn("+0.00", (self.root / "delta.md").read_text())

    def test_explicit_priority_is_dispatched_and_recorded(self):
        self.args.priority = 2
        with patch.object(suite.subprocess, 'run', side_effect=self.fake_run):
            report = suite.run_binary(self.binary, self.root/'priority', self.prompts, self.corpus, self.args)
        self.assertEqual(report['queued_command'][2], '2')
        self.assertEqual(report['queue_priority'], 2)

    def test_invalid_priority_fails_before_launch(self):
        for value in (-1, 4, '2', True):
            with self.subTest(value=value), patch.object(suite.subprocess, 'run') as process:
                self.args.priority = value
                with self.assertRaisesRegex(ValueError, 'priority'):
                    suite.run_binary(self.binary, self.root/'invalid', self.prompts, self.corpus, self.args)
                process.assert_not_called()

    def test_unknown_environment_is_redacted_and_rejected_before_queue(self):
        secret = 'fake-private-api-credential-placeholder'
        with patch.dict(suite.os.environ, {'CRUCIBLE_API_KEY': secret}, clear=True), \
                patch.object(suite.subprocess, 'run') as process, \
                self.assertRaisesRegex(ValueError, 'unknown.*CRUCIBLE_API_KEY'):
            suite.run_binary(self.binary, self.root/'unknown', self.prompts, self.corpus, self.args)
        process.assert_not_called()
        text = (self.root/'unknown/environment.json').read_text()
        self.assertNotIn(secret, text)
        self.assertEqual(json.loads(text)['environment']['CRUCIBLE_API_KEY'], '[REDACTED]')

    def test_environment_policy_mutation_rejects_measurement(self):
        with patch.object(suite.subprocess, 'run', side_effect=self.fake_run), \
                patch.object(suite, 'policy_hashes', return_value={'changed': 'policy'}), \
                self.assertRaisesRegex(ValueError, 'environment policy changed'):
            suite.run_binary(self.binary, self.root/'policy', self.prompts, self.corpus, self.args)
        self.assertFalse((self.root/'policy/results.json').exists())

    def test_worker_policy_mismatch_or_missing_rejects_measurement(self):
        for index, hashes in enumerate((None, {'changed':'policy'})):
            def wrong_worker(command, **kwargs):
                result=self.fake_run(command, **kwargs)
                host=Path(command[-2])
                state=json.loads(host.read_text())
                if hashes is None:state.pop('environment_policy_hashes')
                else:state={**state,'environment_policy_hashes':hashes}
                suite.save(host,state)
                return result
            with self.subTest(hashes=hashes), patch.object(suite.subprocess,'run',side_effect=wrong_worker), \
                    self.assertRaisesRegex(ValueError,'worker environment policy'):
                suite.run_binary(self.binary,self.root/f'worker-policy-{index}',self.prompts,self.corpus,self.args)
            self.assertFalse((self.root/f'worker-policy-{index}/results.json').exists())

    def test_failed_process_is_not_a_result(self):
        with patch.object(suite.subprocess, "run", return_value=subprocess.CompletedProcess([], 124)):
            with self.assertRaisesRegex(ValueError, "failed.*124"):
                suite.run_binary(self.binary, self.root / "a", self.prompts, self.corpus, self.args)
        self.assertFalse((self.root / "a/results.json").exists())

    def test_changed_binary_fails_before_launch(self):
        self.binary.write_bytes(b"changed")
        with patch.object(suite.subprocess, "run") as process:
            with self.assertRaisesRegex(ValueError, "binary differs"):
                suite.run_binary(self.binary, self.root / "a", self.prompts, self.corpus, self.args)
            process.assert_not_called()

    def test_bad_options_and_tokenizer_fail_before_launch(self):
        self.args.tokens = 1
        with self.assertRaisesRegex(ValueError, "tokens"):
            suite.run(self.args)
        self.args.tokens = 3
        (self.checkpoint / "tokenizer.json").write_text("wrong tokenizer")
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            suite.run(self.args)

    def test_budget_and_numeric_arguments(self):
        self.assertLess(suite.check_budget(self.prompts, 160, 1, 1, 100), 270)
        for value in ("nan", "inf", "0", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                suite.positive(value)
        self.assertEqual(suite.positive("125"), 125)


if __name__ == "__main__":
    unittest.main()
