import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import contextlib
import io

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("conversation_run", HERE / "conversation_run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ConversationRunTests(unittest.TestCase):
    def test_fixture_load_and_cache_request(self):
        turns = runner.load_turns(HERE / "conversations/manifest.json", "4k")
        self.assertEqual(len(turns), 3)
        request = runner.request_for(turns[1], "auto", False)
        self.assertEqual(request["cache_prefix_tokens"], 2614)
        self.assertEqual(request["max_tokens"], 1)
        self.assertEqual(request["spec"], 0)
        self.assertNotIn("cache_control", runner.request_for(turns[1], "bypass", True))

    def test_fake_resident_records_first_token_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "engine"
            engine.write_text("#!/usr/bin/env python3\nimport json,sys\n"
                              "print(json.dumps({'event':'ready','context':int(sys.argv[-1])}),flush=True)\n"
                              "for line in sys.stdin:\n"
                              " r=json.loads(line)\n"
                              " print(json.dumps({'event':'token','id':7}),flush=True)\n"
                              " print(json.dumps({'event':'done','prompt_tokens':len(r['ids']),"
                              "'completion_tokens':1,'cache_hit':False,'cached_tokens':0,"
                              "'prefill_tokens':len(r['ids'])}),flush=True)\n")
            engine.chmod(0o755)
            turns = runner.load_turns(HERE / "conversations/manifest.json", "4k")
            result = runner.measure(engine, root, turns, "auto", False, root / "events.jsonl")
            self.assertEqual(len(result["turns"]), 3)
            for turn in result["turns"]:
                self.assertEqual(turn["output_ids"], [7])
                self.assertGreater(turn["ttft_wall_seconds"], 0)
                self.assertEqual(turn["summary"]["cached_tokens"], 0)
            self.assertEqual(len((root / "events.jsonl").read_text().splitlines()), 6)
            self.assertIn("Wall TTFT", runner.markdown({**result, "fixture": "4k"}))
            (root / "config.json").write_text("{}")
            (root / "quantization_config.json").write_text("{}")
            self.check_launch(root, engine)

    def check_launch(self, root, engine):
        gpu = root / "gpu-run"
        gpu.write_text("#!/usr/bin/env python3\nimport os,sys\n"
                       "assert sys.argv[1:5]==['-p','3','-w','3600']\n"
                       "i=sys.argv.index('--')+1\n"
                       "assert sys.argv[i:i+4]==['timeout','-k','10','300']\n"
                       "os.execvp(sys.argv[i],sys.argv[i:])\n")
        gpu.chmod(0o755)
        args = SimpleNamespace(engine=engine, checkpoint=root, fixture="4k", prefill_floor=77,
                               cache_control="auto", legacy_protocol=False, source_commit="fake-cpu",
                               output=root / "out", gpu_run=gpu, gpu_wait=3600)
        corpus = json.loads((HERE / "conversations/manifest.json").read_text())
        actual_digest = runner.digest
        def fake_digest(path):
            if path.name == "tokenizer.json":
                return corpus["tokenizer_sha256"]
            if path.name == corpus["template_file"]:
                return corpus["template_sha256"]
            return actual_digest(path)
        with mock.patch.object(runner, "digest", side_effect=fake_digest), contextlib.redirect_stdout(io.StringIO()):
            runner.launch(args)
        runner.worker(root / "out/request.json")
        report = json.loads((root / "out/results.json").read_text())
        self.assertEqual(report["turns"][2]["output_ids"], [7])
        self.assertEqual(json.loads((root / "out/exit.json").read_text())["returncode"], 0)
        self.assertTrue((root / "out/host.json").exists())

    def test_corrupt_fixture_hash_fails(self):
        with mock.patch.object(runner, "digest", return_value="bad"):
            with self.assertRaisesRegex(ValueError, "messages hash differs"):
                runner.load_turns(HERE / "conversations/manifest.json", "4k")


if __name__ == "__main__":
    unittest.main()
