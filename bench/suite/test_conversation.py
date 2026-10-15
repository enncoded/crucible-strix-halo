import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock
import tempfile
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("conversation_generate", HERE / "conversation_generate.py")
conversation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conversation)


class ConversationTests(unittest.TestCase):
    def test_committed_fixtures_and_stable_prefixes(self):
        root = HERE / "conversations"
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(len(manifest["scenarios"]), 3)
        for scenario in manifest["scenarios"]:
            previous = []
            for turn in scenario["turns"]:
                ids_path = root / turn["ids_file"]
                messages_path = root / turn["messages_file"]
                self.assertEqual(conversation.hashed(ids_path), turn["ids_sha256"])
                self.assertEqual(conversation.hashed(messages_path), turn["messages_sha256"])
                ids = [int(value) for value in ids_path.read_text().strip().split(",")]
                self.assertEqual(len(ids), turn["tokens"])
                self.assertEqual(conversation.common_prefix(previous, ids), turn["shared_prefix_tokens"])
                self.assertEqual(turn["tokens"] - turn["cache_prefix_tokens"], 7)
                if previous:
                    self.assertTrue(1400 < turn["new_tokens"] < 1600)
                raw = json.loads(messages_path.read_text())
                self.assertEqual({m["role"] for m in raw["messages"]}, {"system", "user", "assistant", "tool"})
                previous = ids

    def test_generator_main_with_cpu_template(self):
        class Template:
            special_tokens = {}
            def __init__(self, checkpoint):
                self.template = self
            def render(self, messages, add_generation_prompt, **kwargs):
                return json.dumps(messages) + ("[GEN]" if add_generation_prompt else "")
        class Tokenizer:
            @staticmethod
            def from_file(path):
                return Tokenizer()
            def encode(self, text, **kwargs):
                return SimpleNamespace(ids=[ord(char) for char in text])
        modules = {"server.template": SimpleNamespace(ChatTemplate=Template),
                   "tokenizers": SimpleNamespace(Tokenizer=Tokenizer)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tokenizer.json").write_text("{}")
            (root / "chat_template.jinja").write_text("cpu test template")
            argv = ["generate", str(root), "--output", str(root / "out")]
            with mock.patch.dict(sys.modules, modules), mock.patch.object(sys, "argv", argv):
                conversation.main()
            manifest = json.loads((root / "out/manifest.json").read_text())
            self.assertEqual(len(manifest["scenarios"]), 3)
            self.assertTrue(manifest["scripted_history"])
            for scenario in manifest["scenarios"]:
                self.assertEqual(len(scenario["turns"]), 3)
                self.assertTrue(all(turn["cache_prefix_tokens"] == turn["tokens"] - 5 for turn in scenario["turns"]))

    def test_common_prefix(self):
        self.assertEqual(conversation.common_prefix([1, 2], [1, 2, 3]), 2)
        self.assertEqual(conversation.common_prefix([1, 2], [1, 3]), 1)
        self.assertEqual(conversation.common_prefix([], [1]), 0)

    def test_fill_keeps_messages_and_uses_complete_code_records(self):
        prefix = conversation.base_messages()
        encode = lambda messages: list(range(sum(len(str(message)) for message in messages)))
        messages, ids = conversation.fill_turn(prefix, 3000, 0, encode)
        self.assertEqual(messages[:len(prefix)], prefix)
        self.assertEqual(messages[-2]["role"], "tool")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertLess(abs(len(ids) - 3000), 200)
        self.assertEqual(len(prefix), 3)
        later = conversation.extension(messages, 1)
        self.assertEqual(later[:len(messages)], messages)
        self.assertEqual(later[-1]["tool_calls"][0]["function"]["name"], "run_tests")


if __name__ == "__main__":
    unittest.main()
