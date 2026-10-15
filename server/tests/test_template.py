import json
from pathlib import Path
import tempfile
import unittest
from server.template import ChatTemplate
from server.api import ApiError, chat_messages

FIXTURE = Path(__file__).parent / "fixtures/qwen_chat_template.jinja"

class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": FIXTURE.read_text()}))
        self.template = ChatTemplate(root)

    def tearDown(self):
        self.temp.cleanup()

    def render(self, messages, **options):
        return self.template.render(chat_messages(messages), **options)

    def test_thinking_off(self):
        self.assertEqual(self.render([dict(role="user", content="hello")]),
                         "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def test_system_and_trim(self):
        rendered = self.render([dict(role="system", content="  rules  "), dict(role="user", content=" hello ")])
        self.assertTrue(rendered.startswith("<|im_start|>system\nrules<|im_end|>\n<|im_start|>user\nhello"))

    def test_multi_turn(self):
        rendered = self.render([dict(role="user", content="first"), dict(role="assistant", content="answer"),
                                dict(role="user", content="second")])
        self.assertIn("<|im_start|>assistant\n<think>\n\n</think>\n\nanswer<|im_end|>\n", rendered)

    def test_stable_cache_boundary(self):
        messages = [dict(role="user", content="first")]
        prefix = self.render(messages, add_generation_prompt=False)
        request = self.render(messages)
        next_turn = self.render(messages + [dict(role="assistant", content="answer"),
                                             dict(role="user", content="second")])
        self.assertTrue(request.startswith(prefix))
        self.assertTrue(next_turn.startswith(prefix))
        self.assertFalse(prefix.endswith("<think>\n\n</think>\n\n"))

    def test_thinking_on(self):
        rendered = self.render([dict(role="user", content="hello")], enable_thinking=True)
        self.assertIn("Reasoning effort is set to xhigh.", rendered)
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n<think>\n"))

    def test_medium_thinking(self):
        rendered = self.render([dict(role="system", content="rules"), dict(role="user", content="hello")],
                               enable_thinking=True, reasoning_effort="medium")
        self.assertNotIn("Reasoning effort", rendered)
        self.assertTrue(rendered.endswith("<think>\n"))

    def test_empty_assistant_prefix(self):
        rendered = self.render([dict(role="user", content="hello"), dict(role="assistant", content="")])
        self.assertEqual(rendered.count("<|im_start|>assistant"), 2)
        self.assertTrue(rendered.endswith("<think>\n\n</think>\n\n"))

    def test_text_blocks_and_reasoning(self):
        rendered = self.render([dict(role="user", content=[dict(type="text", text="a"), dict(type="text", text="b")]),
                                dict(role="assistant", content="answer", reasoning_content="note")])
        self.assertIn("user\nab<|im_end|>", rendered)
        self.assertIn("<think>\nnote\n</think>\n\nanswer", rendered)

if __name__ == "__main__":
    unittest.main()
