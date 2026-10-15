import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


run = module("run")
generate = module("generate")
gates = module("gate_check")


# CRUCIBLE_PI_AI_DIR and CRUCIBLE_TEST_CKPT only enable optional server tests in the full `make test-cpu`
# environment. The runner refuses unknown CRUCIBLE_* names, so clear them here instead of widening the allow-list.
_TEST_DEPENDENCY_ENV = patch.dict(os.environ)


def setUpModule():
    _TEST_DEPENDENCY_ENV.start()
    for name in ("CRUCIBLE_PI_AI_DIR", "CRUCIBLE_TEST_CKPT"):
        os.environ.pop(name, None)


def tearDownModule():
    _TEST_DEPENDENCY_ENV.stop()


class CliTests(unittest.TestCase):
    def test_build_records_identity_and_hashes_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Parents from the file paths, so that the fixture follows a layout move.
            for name, text in (("engine/cli/gen_cli.h", "GenResult RunGen(Engine& e);"),
                               ("engine/exl3/safetensors.h", "// fixture")):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text(text)
            for name in ("engine", "kernels", "bench"):
                (root / name).mkdir(exist_ok=True)
            args = argparse.Namespace(source=root, output=root / "bin/resident", hipcc="fake-hipcc",
                                      source_commit="abc", source_branch="fixture")
            def compile_only(command, check):
                self.assertIn("--offload-arch=gfx1151", command)
                args.output.write_bytes(b"compiled fixture")
            with patch.object(run.subprocess, "check_output", return_value="fake compiler"), \
                    patch.object(run.subprocess, "run", side_effect=compile_only):
                run.build(args)
            metadata = json.loads(Path(str(args.output) + ".build.json").read_text())
            self.assertEqual(metadata["source_commit"], "abc")
            self.assertEqual(metadata["binary_sha256"], run.digest(args.output))
            self.assertEqual(len(metadata["source_hashes"]), 2)
            args.source_branch = None
            with self.assertRaisesRegex(ValueError, "both"):
                run.build(args)
            (root / "engine/cli/gen_cli.h").write_text("// unsupported")
            with self.assertRaisesRegex(ValueError, "RunGen"):
                run.build(args)

    def test_cli_dispatch_and_friendly_error(self):
        with patch.object(run.sys, "argv", ["run.py", "build", "--source", ".", "--output", "x"]), \
                patch.object(run, "build") as build:
            run.main()
            build.assert_called_once()
        args = ["run.py", "run", ".", "--binary", "binary", "--output", "results"]
        with patch.object(run.sys, "argv", args), patch.object(run, "run") as command:
            run.main()
            command.assert_called_once()
        with patch.object(run.sys, "argv", args), patch.object(run, "run", side_effect=ValueError("bad test")), \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as stopped:
                run.main()
            self.assertEqual(stopped.exception.code, 1)
            self.assertIn("bad test", errors.getvalue())

    def test_run_cli_priority_default_and_range(self):
        base = ['run.py', 'run', '.', '--binary', 'binary', '--output', 'results']
        for value in (None, 0, 1, 2, 3):
            argv = base if value is None else [*base, '--priority', str(value)]
            with patch.object(run.sys, 'argv', argv), patch.object(run, 'run') as execute:
                run.main()
                self.assertEqual(execute.call_args.args[0].priority, 3 if value is None else value)
        for value in ('-1', '4', 'nan'):
            with patch.object(run.sys, 'argv', [*base, '--priority', value]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                run.main()
            self.assertEqual(stopped.exception.code, 2)

    def test_worker_snapshot_precedes_exec_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "host.json"
            argv = ["run.py", "_worker", "/fake", "ckpt", "requests", "4096", str(path)]
            with patch.object(run.sys, "argv", argv), patch.object(run.os, "execv") as execute, \
                    patch.dict(run.os.environ, {"HSA_USERPTR_FOR_PAGED_MEM": "0"}):
                run.worker()
                execute.assert_called_once_with("/fake", ["/fake", "ckpt", "requests", "4096"])
            state = json.loads(path.read_text())
            self.assertIn("MemAvailable", state["meminfo"])
            self.assertIn("buddyinfo", state)
            self.assertEqual(state["environment"]["HSA_USERPTR_FOR_PAGED_MEM"], "0")

    def test_worker_redacts_unknown_environment_and_refuses_exec(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'host.json'
            argv = ['run.py', '_worker', '/fake', 'ckpt', 'requests', '4096', str(path)]
            secret = 'fake-secret-looking-value-do-not-publish'
            with patch.object(run.sys, 'argv', argv), patch.object(run.os, 'execv') as execute, \
                    patch.dict(run.os.environ, {'CRUCIBLE_API_KEY': secret}, clear=True), \
                    self.assertRaisesRegex(ValueError, 'unknown.*CRUCIBLE_API_KEY'):
                run.worker()
            execute.assert_not_called()
            state = json.loads(path.read_text())
            self.assertEqual(state['environment']['CRUCIBLE_API_KEY'], '[REDACTED]')
            self.assertEqual(state['unknown_crucible_keys'], ['CRUCIBLE_API_KEY'])
            self.assertNotIn(secret, path.read_text())

    def test_worker_records_central_allowlisted_runtime_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'host.json'
            argv = ['run.py', '_worker', '/fake', 'ckpt', 'requests', '4096', str(path)]
            env = {'CRUCIBLE_COMPACT_STATE': '1', 'HSA_USE_SVM': '0',
                   'GPU_PINNED_MIN_XFER_SIZE': '4096'}
            with patch.object(run.sys, 'argv', argv), patch.object(run.os, 'execv'), \
                    patch.dict(run.os.environ, env, clear=True):
                run.worker()
            state = json.loads(path.read_text())
            for name, value in env.items(): self.assertEqual(state['environment'][name], value)
            self.assertEqual(state['unknown_crucible_keys'], [])
            self.assertEqual(len(state['environment_policy_hashes']), 2)

    def test_generator_trim_and_tokenizer_invocation(self):
        self.assertEqual(generate.trim_middle(list(range(10)), 6, 2), [0, 1, 2, 3, 8, 9])
        for length in (1, 20):
            with self.assertRaises(ValueError):
                generate.trim_middle(list(range(10)), length, 2)
        original = list(generate.sys.argv)
        def fake_helper(path, run_name):
            self.assertEqual(generate.sys.argv[-1], "--chat")
            self.assertEqual(run_name, "__main__")
            print("1,2,3")
        with patch.object(generate.runpy, "run_path", side_effect=fake_helper):
            self.assertEqual(generate.encode(Path("checkpoint"), "hello"), [1, 2, 3])
        self.assertEqual(generate.sys.argv, original)

    def test_generator_writes_manifest_and_long_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tools").mkdir()
            (root / "tools/tok.py").write_text("# fixture")
            (root / "tokenizer.json").write_text("{}")
            source = {"short": [{"name": "short", "category": "chat", "text": "hi"}],
                      "long": {"prefix": "start", "record": "record", "suffix": "finish", "lengths": [64, 128]}}
            (root / "prompts.json").write_text(json.dumps(source))
            with patch.object(generate, "SUITE", root), patch.object(generate, "ROOT", root), \
                    patch.object(generate.sys, "argv", ["generate.py", str(root)]), \
                    patch.object(generate, "encode", side_effect=lambda ckpt, text, chat=True: list(range(len(text)))):
                generate.main()
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual([p["tokens"] for p in manifest["prompts"]], [2, 64, 128])

    def test_gate_cli_fails_closed(self):
        with patch.object(gates.argparse.ArgumentParser, "parse_args", return_value=argparse.Namespace(
                baseline=Path("a"), one=Path("b"), eight=Path("c"), sweep=Path("d"))), \
                patch.object(gates, "check_dumps", side_effect=ValueError("fixture failure")), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                gates.main()
            self.assertEqual(stopped.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
