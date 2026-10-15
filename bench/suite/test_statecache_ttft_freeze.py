import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import statecache_ttft as tool


class FrozenTimingInputTests(unittest.TestCase):
    def run_mutation(self, when):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = root / "fixtures"
            fixtures.mkdir()
            engine = root / "engine"
            engine.write_text("fake engine")
            for name in ("tokenizer.json", "chat_template.jinja", "config.json", "quantization_config.json", "model.safetensors.index.json"):
                (root / name).write_text(name)
            ids = fixtures / "ids.txt"
            messages = fixtures / "messages.json"
            ids.write_text("1,2,3")
            messages.write_text("[]")
            manifest_path = fixtures / "manifest.json"
            manifest = dict(tokenizer_sha256=tool.digest(root / "tokenizer.json"),
                            template_sha256=tool.digest(root / "chat_template.jinja"), template_file="chat_template.jinja",
                            scenarios=[dict(name="tiny", turns=[dict(turn=0, tokens=3, cache_prefix_tokens=2,
                                ids_file=ids.name, ids_sha256=tool.digest(ids), messages_file=messages.name,
                                messages_sha256=tool.digest(messages))])])
            original = json.dumps(manifest).encode()
            manifest_path.write_bytes(original)
            observed = []
            process = SimpleNamespace(stdin=io.StringIO(), wait=Mock(), terminate=Mock())

            def mutate():
                ids.write_text("4,5,6")
                manifest["scenarios"][0]["turns"][0]["ids_sha256"] = tool.digest(ids)
                manifest_path.write_text(json.dumps(manifest))

            def launch(*args, **kwargs):
                if when == "startup":
                    mutate()
                return process

            def request(process, values, boundary, control):
                observed.append((control, tuple(values), boundary))
                if when == "between" and control == "bypass":
                    mutate()
                return dict(ttft_seconds=.01, first_id=7, prefill_tokens=3, cached_tokens=0)

            output = root / "report.json"
            argv = ["statecache_ttft.py", str(engine), str(root), str(fixtures), str(output)]
            with patch.object(tool.sys, "argv", argv), patch.object(tool.subprocess, "Popen", side_effect=launch), \
                 patch.object(tool, "event", return_value=dict(event="ready")), \
                 patch.object(tool, "request", side_effect=request), contextlib.redirect_stdout(io.StringIO()):
                tool.main()
            result = json.loads(output.read_text())
            self.assertEqual(observed, [("bypass", (1, 2, 3), 2), ("auto", (1, 2, 3), 2)])
            self.assertEqual(result["manifest"], json.loads(original))
            self.assertEqual(result["artifacts"]["manifest_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(result["status"], "PASS")

    def test_main_freezes_inputs_before_process_start(self):
        self.run_mutation("startup")

    def test_main_reuses_inputs_after_bypass(self):
        self.run_mutation("between")


if __name__ == "__main__":
    unittest.main()
