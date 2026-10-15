"""CPU tests for tools/agent_soak.py: the soak rule, synthetic content, and stream parsing (no server)."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import agent_soak as soak

# The long session of the Oct 5 incident (the logged numbers: prompt, cached), the served build, replay off.
INCIDENT = [(12834, 0), (17510, 12288), (22749, 16384), (23468, 20480), (24994, 20480), (25558, 24576),
            (34299, 32768), (54752, 32768), (74721, 53248), (94222, 53248), (113374, 53248), (219071, 53248)]


def turns(pairs, completion=300, session="long"):
    return [dict(session=session, turn=i, prompt_tokens=p, cached_tokens=c, completion_tokens=completion)
            for i, (p, c) in enumerate(pairs)]


class RuleTest(unittest.TestCase):
    def test_incident_fails_from_94222(self):
        report = soak.check(turns(INCIDENT))
        self.assertEqual(report["verdict"], "FAIL")
        first = report["failures"][0]
        self.assertEqual((first["prompt_tokens"], first["prefilled"], first["allowed"]), (94222, 40974, 19501 + 4096))

    def test_fixed_pattern_passes(self):
        fixed = [(p, 0 if i == 0 else (INCIDENT[i - 1][0] - 4) // 4096 * 4096) for i, (p, _) in enumerate(INCIDENT)]
        self.assertEqual(soak.check(turns(fixed))["verdict"], "PASS")

    def test_sessions_are_independent(self):
        rows = turns([(12000, 0), (30000, 8192)], session="a") + turns([(12000, 0), (13000, 8192)], session="b")
        report = soak.check(rows)
        self.assertEqual(report["verdict"], "PASS")  # b re-prefills 4808 of 1000 appended + 4096
        self.assertEqual(report["sessions"], {"a": 2, "b": 2})

    def test_replay_rule(self):
        # The previous turn ended at 20000 + 300 generated; a replay hit restores 20299 of the next 21000.
        hit = turns([(20000, 0), (21000, 20299)])
        self.assertEqual(soak.check(hit, expect_replay=True)["verdict"], "PASS")
        frontier = turns([(20000, 0), (21000, 16384)])
        self.assertEqual(soak.check(frontier)["verdict"], "PASS")
        report = soak.check(frontier, expect_replay=True)
        self.assertEqual(report["verdict"], "FAIL")
        self.assertEqual(report["failures"][0]["rule"], "replay")

    def test_cold_first_turn_may_omit_the_cache_count(self):
        # Halogen 0.16.2 omits cached_tokens on a cold first turn; a later turn without it is still an error.
        rows = turns([(20000, 0), (21000, 20000)])
        rows[0]["cached_tokens"] = None
        self.assertEqual(soak.check(rows)["verdict"], "PASS")

    def test_server_error_is_named(self):
        rows = turns([(20000, 0), (21000, 20000)])
        rows[1].update(prompt_tokens=None, cached_tokens=None, completion_tokens=None, error="Malformed model tool call")
        report = soak.check(rows)
        self.assertEqual(report["verdict"], "ERROR")
        self.assertIn("Malformed model tool call", report["error"])

    def test_missing_cache_count_is_an_error(self):
        rows = turns([(20000, 0), (21000, 0)])
        rows[1]["cached_tokens"] = None
        self.assertEqual(soak.check(rows)["verdict"], "ERROR")


class ContentTest(unittest.TestCase):
    def test_deterministic_and_sized(self):
        a, b = soak.tool_result(7, 2000), soak.tool_result(7, 2000)
        self.assertEqual(a, b)
        self.assertNotEqual(a, soak.tool_result(8, 2000))
        self.assertAlmostEqual(len(a), 2000 * soak.CHARS_PER_TOKEN, delta=200)

    def test_tools_and_system(self):
        tools = soak.tool_specs()
        self.assertEqual(len(tools), 23)
        self.assertEqual(len({tool["function"]["name"] for tool in tools}), 23)
        self.assertGreater(len(soak.system_prompt(10000)), 9000 * soak.CHARS_PER_TOKEN)

    def test_schedule_reaches_the_target(self):
        sizes = soak.schedule("long", 220000, 12000)
        self.assertGreaterEqual(12000 + sum(sizes), 200000)
        self.assertTrue(all(500 <= size <= 25000 for size in sizes))
        self.assertLess(12000 + sum(soak.schedule("short", 40000, 12000)), 45000)


class StreamTest(unittest.TestCase):
    def test_stable_ids_and_hash(self):
        reply = dict(content="c", reasoning="", tool_calls=[dict(id="call_9f", type="function",
                                                                  function=dict(name="read", arguments="{}"))])
        fixed = soak.stable_call_ids(reply, "agent-0_3")
        self.assertEqual(fixed["tool_calls"][0]["id"], "call_agent-0_3_0")
        self.assertEqual(reply["tool_calls"][0]["id"], "call_9f")  # the reply itself is unchanged
        self.assertEqual(soak.content_hash(reply), soak.content_hash(fixed))

    def test_agent_schedule(self):
        sizes = soak.schedule("agent", 44000, 9000 + 24000)
        self.assertTrue(sizes and max(sizes) < 10000 and 9000 + 24000 + sum(sizes) <= 44000)
        self.assertEqual(soak.AGENT_SCENARIOS["agents"][0], 4)
        self.assertIn("--streams 4 --ctx 131072", soak.AGENT_SCENARIOS["agents"][1])
        self.assertEqual(soak.AGENT_SCENARIOS["served3"], (3, soak.SERVED))

    def test_parse_stream(self):
        chunks = [dict(choices=[dict(delta=dict(reasoning_content="think "))]),
                  dict(choices=[dict(delta=dict(reasoning_content="more"))]),
                  dict(choices=[dict(delta=dict(tool_calls=[dict(index=0, id="c1", type="function",
                                                                 function=dict(name="read", arguments='{"pa'))]))]),
                  dict(choices=[dict(delta=dict(tool_calls=[dict(index=0, function=dict(arguments='th":"a"}'))]))]),
                  dict(choices=[dict(delta={}, finish_reason="tool_calls")]),
                  dict(choices=[], usage=dict(prompt_tokens=100, completion_tokens=9,
                                              prompt_tokens_details=dict(cached_tokens=64)))]
        lines = [b"data: " + json.dumps(chunk).encode() for chunk in chunks] + [b"", b"data: [DONE]"]
        reply = soak.parse_stream(iter(lines))
        self.assertEqual(reply["reasoning"], "think more")
        self.assertEqual(reply["tool_calls"], [dict(id="c1", type="function",
                                                     function=dict(name="read", arguments='{"path":"a"}'))])
        self.assertEqual((reply["prompt_tokens"], reply["cached_tokens"], reply["completion_tokens"]), (100, 64, 9))
        self.assertEqual(reply["finish"], "tool_calls")

    def test_error_chunk(self):
        # Crucible sends an error chunk after HTTP 200 when the reply fails late (soak run Oct 5, turn 4).
        lines = [b'data: {"choices":[{"delta":{"content":"x"}}]}', b'',
                 b'data: {"error":{"message":"Malformed model tool call: x","type":"server_error"}}', b'',
                 b"data: [DONE]"]
        reply = soak.parse_stream(iter(lines))
        self.assertEqual(reply["error"], "Malformed model tool call: x")
        self.assertIsNone(reply["prompt_tokens"])

    def test_request_body(self):
        args = SimpleNamespace(max_tokens=384, thinking="off", reasoning_effort="high", thinking_budget=None)
        body = soak.request_body("m", [], [], args)
        self.assertEqual(body["chat_template_kwargs"], dict(enable_thinking=False))
        self.assertTrue(body["stream"] and body["stream_options"]["include_usage"])
        self.assertNotIn("reasoning_effort", body)
        body = soak.request_body("m", [], [], SimpleNamespace(max_tokens=None, thinking="on", reasoning_effort="high",
                                                               thinking_budget=128))
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertNotIn("chat_template_kwargs", body)
        # Pi's max_tokens: a small one makes the answer room close the think block after one token.
        self.assertEqual((body["max_tokens"], body["max_thinking_tokens"]), (32768, 128))

    def test_next_messages(self):
        reply = dict(content="", reasoning="r", tool_calls=[dict(id="c1", type="function",
                                                                  function=dict(name="read", arguments="{}"))])
        added = soak.next_messages(reply, "RESULT")
        self.assertEqual(added[0]["role"], "assistant")
        self.assertEqual(added[0]["reasoning_content"], "r")
        self.assertEqual(added[1], dict(role="tool", tool_call_id="c1", content="RESULT"))
        # agents/served3: the client drops the reasoning, so a think turn's tail never repeats (Oct 6 failure).
        self.assertNotIn("reasoning_content", soak.next_messages(reply, "RESULT", keep_reasoning=False)[0])
        plain = soak.next_messages(dict(content="done", reasoning="", tool_calls=[]), "RESULT")
        self.assertEqual([m["role"] for m in plain], ["assistant", "user"])
        self.assertIn("RESULT", plain[1]["content"])


class EnvironmentTest(unittest.TestCase):
    def test_the_report_records_the_switch_environment(self):
        # Train 19: the gate checks that the short soaks ran with CRUCIBLE_PF_LAUNCH_ROWS=4096.
        env = {"CRUCIBLE_PF_LAUNCH_ROWS": "4096", "CRUCIBLE_MTP_SIDECAR": "/x", "PATH": "/bin"}
        self.assertEqual(soak.switch_environment(env),
                         {"CRUCIBLE_PF_LAUNCH_ROWS": "4096", "CRUCIBLE_MTP_SIDECAR": "/x"})
        source = Path(soak.__file__).read_text()
        self.assertIn("environment=switch_environment(os.environ)", source)

    def test_the_report_records_the_sha256_of_the_engine_it_started(self):
        # Review t19s1b: a report from another build at the same path must not pass the gate.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "crucible-t19"
            engine.write_bytes(b"binary")
            self.assertEqual(soak.file_sha256(engine), hashlib.sha256(b"binary").hexdigest())
            self.assertIsNone(soak.file_sha256(None))
            self.assertIsNone(soak.file_sha256(Path(tmp) / "missing"))
        source = Path(soak.__file__).read_text()
        self.assertIn("engine_sha256=file_sha256(args.engine)", source)


if __name__ == "__main__":
    unittest.main()
