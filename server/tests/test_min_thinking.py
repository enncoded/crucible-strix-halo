"""--min-thinking-tokens: a thinking request whose answer-room close limit is below N is served with thinking off."""
import json
import unittest
from dataclasses import replace

from server.api import ServerDefaults
from server.tests.serve_fixtures import chunks, invoke
from server.tests.test_server_defaults import parse
from server.tests.test_think_budget import (ANSWER, IM_END, PROMPT, REASON, BudgetTokenizer, SequenceBackend,
                                            chat)
from server.think_budget import skips_thinking

ON, OFF = [1, 1], [1, 1, 1, 1, 1]  # the thinking-off render has a longer prompt, as the empty think block does


class RenderTokenizer(BudgetTokenizer):
    """Encodes the thinking-on prompt shorter than the thinking-off prompt, like the real chat template."""

    def encode(self, request):
        return PROMPT + (ON if request.enable_thinking else OFF)


def run(backend, body, **service):
    response = invoke(backend, RenderTokenizer(), body, service_options=service or None)
    return response, (chunks(response) if body.get("stream") else json.loads(response.body))


def answer_backend():
    return SequenceBackend(([ANSWER, IM_END], "stop"))


class RuleTest(unittest.TestCase):
    def test_the_rule_on_the_close_limit(self):
        # limit = max_tokens - room, before the max(1, ...)
        self.assertTrue(skips_thinking(256, None, None, 256))
        self.assertTrue(skips_thinking(1100, None, None, 256))
        self.assertTrue(skips_thinking(1279, None, None, 256))  # limit 255
        self.assertFalse(skips_thinking(1280, None, None, 256))  # limit 256
        self.assertTrue(skips_thinking(10, None, 5, 256))  # an explicit room of 5 leaves 5

    def test_off_switches(self):
        self.assertFalse(skips_thinking(256, None, None, 0))  # --min-thinking-tokens 0
        self.assertFalse(skips_thinking(256, 50, None, 256))  # an explicit thinking budget
        self.assertFalse(skips_thinking(256, None, 0, 256))  # --thinking-answer-room 0: the room is 0


class SkipTest(unittest.TestCase):
    def test_a_small_max_tokens_serves_thinking_off(self):
        backend = answer_backend()
        _, body = run(backend, chat(max_tokens=256), min_thinking_tokens=256)
        (ids, request, options), = backend.submissions
        self.assertEqual(ids, PROMPT + OFF)  # the enable_thinking: false render
        self.assertFalse(request.enable_thinking)
        self.assertEqual(request.max_tokens, 256)  # the answer gets all of max_tokens
        self.assertNotIn("think_cap", options)
        self.assertEqual(body["usage"]["completion_tokens_details"], dict(reasoning_skipped_by="min_thinking_tokens"))
        self.assertEqual(body["choices"][0]["message"]["content"], "ok")
        self.assertNotIn("reasoning_content", body["choices"][0]["message"])

    def test_the_skipped_request_is_the_thinking_off_request(self):
        skipped, off = answer_backend(), answer_backend()
        run(skipped, chat(max_tokens=300, seed=3), min_thinking_tokens=256)
        run(off, chat(max_tokens=300, seed=3, enable_thinking=False), min_thinking_tokens=256)
        self.assertEqual(skipped.submissions[0][0], off.submissions[0][0])
        self.assertEqual(replace(skipped.submissions[0][1], thinking_skipped_by=None), off.submissions[0][1])
        self.assertEqual(skipped.submissions[0][2], off.submissions[0][2])

    def test_stream_usage_names_the_skip(self):
        backend = answer_backend()
        _, parts = run(backend, chat(max_tokens=256, stream=True, stream_options=dict(include_usage=True)),
                       min_thinking_tokens=256)
        usage = [part["usage"] for part in parts if part.get("usage")]
        self.assertEqual(usage[-1]["completion_tokens_details"], dict(reasoning_skipped_by="min_thinking_tokens"))
        self.assertFalse(any("reasoning_content" in choice["delta"] for part in parts for choice in part["choices"]))

    def test_1100_skips_and_1280_thinks(self):
        backend = answer_backend()
        run(backend, chat(max_tokens=1100), min_thinking_tokens=256)
        self.assertFalse(backend.submissions[0][1].enable_thinking)
        backend = SequenceBackend(([REASON] * 3, "length"))
        run(backend, chat(max_tokens=1280), min_thinking_tokens=256)
        self.assertTrue(backend.submissions[0][1].enable_thinking)
        self.assertEqual(backend.submissions[0][2]["think_cap"], 256)

    def test_flag_zero_gives_todays_close_at_one_token(self):
        backend = SequenceBackend(([REASON], "length"), ([ANSWER, IM_END], "stop"))
        _, body = run(backend, chat(max_tokens=256), min_thinking_tokens=0)
        self.assertEqual(backend.submissions[0][2]["think_cap"], 1)
        self.assertEqual(backend.submissions[0][0], PROMPT + ON)
        self.assertEqual(body["usage"]["completion_tokens_details"]["reasoning_closed_by"], "answer_room")
        self.assertNotIn("reasoning_skipped_by", body["usage"]["completion_tokens_details"])


class ExactnessTest(unittest.TestCase):
    """A request that the rule does not touch gives the same job as with the flag at 0."""

    def jobs(self, body, **service):
        out = []
        for minimum in (256, 0):
            backend = SequenceBackend(([REASON] * 3, "length"), ([ANSWER, IM_END], "stop"))
            run(backend, dict(body, seed=7), min_thinking_tokens=minimum, **service)  # a fixed seed: the same Request
            out.append(backend.submissions)
        return out

    def test_a_large_max_tokens(self):
        with_rule, without = self.jobs(chat(max_tokens=8192))
        self.assertEqual(with_rule, without)
        self.assertEqual(with_rule[0][2]["think_cap"], 8192 - 1228)

    def test_thinking_off(self):
        with_rule, without = self.jobs(chat(max_tokens=256, enable_thinking=False))
        self.assertEqual(with_rule, without)

    def test_an_explicit_request_budget(self):
        with_rule, without = self.jobs(chat(max_tokens=256, max_thinking_tokens=100))
        self.assertEqual(with_rule, without)
        self.assertTrue(with_rule[0][1].enable_thinking)
        self.assertEqual(with_rule[0][2]["think_cap"], 1)  # the answer room still limits it, as before

    def test_a_server_thinking_budget(self):
        with_rule, without = self.jobs(chat(max_tokens=256), max_thinking_tokens=100)
        self.assertEqual(with_rule, without)
        self.assertTrue(with_rule[0][1].enable_thinking)

    def test_room_zero(self):
        with_rule, without = self.jobs(chat(max_tokens=256), thinking_answer_room=0)
        self.assertEqual(with_rule, without)
        self.assertNotIn("think_cap", with_rule[0][2])


class FinalCountTest(unittest.TestCase):
    """--default-max-tokens max sets max_tokens in Service.fit from the thinking-on prompt, and the skip changes the
    prompt. The decision uses the fitted count; a skip then fits again on the thinking-off prompt."""

    def test_max_decides_with_the_context_room(self):
        backend = answer_backend()
        backend.context = 1100  # room with thinking on: 1100 - 5 + 1 = 1096; the close limit is 72
        run(backend, chat(), defaults=ServerDefaults(max_tokens="max"), min_thinking_tokens=256)
        ids, request, _ = backend.submissions[0]
        self.assertFalse(request.enable_thinking)
        self.assertEqual(ids, PROMPT + OFF)
        self.assertEqual(request.max_tokens, 1100 - len(ids) + 1)  # fitted again: the off prompt is 3 tokens longer
        self.assertFalse(request.max_tokens_all)

    def test_max_with_a_large_room_keeps_thinking(self):
        backend = SequenceBackend(([REASON] * 3, "length"))
        backend.context = 4000
        run(backend, chat(), defaults=ServerDefaults(max_tokens="max"), min_thinking_tokens=256)
        ids, request, options = backend.submissions[0]
        self.assertTrue(request.enable_thinking)
        self.assertEqual((ids, request.max_tokens, options["think_cap"]),
                         (PROMPT + ON, 4000 - len(ids) + 1, 4000 - len(ids) + 1 - 1024))

    def test_max_at_the_edge_uses_the_thinking_on_room(self):
        # The close limit is 256 when the room is 1280. The off prompt would give 3 fewer tokens, and that must not
        # change a decision that was made on the thinking-on prompt.
        backend = SequenceBackend(([REASON] * 3, "length"))
        backend.context = 1280 + len(PROMPT + ON) - 1
        run(backend, chat(), defaults=ServerDefaults(max_tokens="max"), min_thinking_tokens=256)
        self.assertTrue(backend.submissions[0][1].enable_thinking)

    def test_clamp_decides_with_the_clamped_count(self):
        backend = answer_backend()
        backend.context = 1100
        run(backend, chat(max_tokens=32768), clamp_max_tokens=True, min_thinking_tokens=256)
        ids, request, _ = backend.submissions[0]
        self.assertFalse(request.enable_thinking)
        self.assertEqual(request.max_tokens, 1100 - len(ids) + 1)


class FlagTest(unittest.TestCase):
    def test_default_is_256(self):
        options, _ = parse([])
        self.assertEqual(options["min_thinking_tokens"], 256)

    def test_values_reach_the_service(self):
        for value in ("0", "1", "262144"):
            options, _ = parse(["--min-thinking-tokens", value])
            self.assertEqual(options["min_thinking_tokens"], int(value))

    def test_bad_values_exit_2(self):
        for value in ("-1", "262145", "x", "1.5", ""):
            with self.subTest(value=value):
                code, _ = parse(["--min-thinking-tokens", value])
                self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
