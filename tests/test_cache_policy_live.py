import unittest
from bench import cache_policy_live as live


def capture(tokens, kind, evicted=0, superseded=0):
    return (f"cache_capture tokens={tokens} class={kind} saved=1 evicted={evicted} "
            f"superseded_evicted={superseded} cache_bytes=1")


LOG = ["cache_policy_step a1", capture(8192, "shared"), capture(12288, "session"),
       "cache_policy_step b1", capture(12288, "session"),
       "cache_policy_step a2", capture(16384, "session", 1, 1),
       "cache_policy_step b2", "unrelated engine line", capture(16384, "session", 1, 1),
       "cache_policy_step c1", capture(12288, "session", 1, 0),
       "cache_policy_step references"]


def rows(c1=8192):
    cached = dict(a1=(0, None), b1=(8192, "shared"), a2=(12288, "session"), b2=(12288, "session"),
                  c1=(c1, "shared" if c1 else None))
    result = {name: dict(ids=[1, 2], cached_tokens=tokens, cache_class=kind) for name, (tokens, kind) in cached.items()}
    for name in live.REFERENCES:
        result[f"{name}_ref"] = dict(ids=[1, 2], cached_tokens=0, cache_class=None)
    return result


class CachePolicyLiveTest(unittest.TestCase):
    def test_sessions_share_only_the_system_block(self):
        a1, a2, b1 = live.session_ids("a", 1), live.session_ids("a", 2), live.session_ids("b", 1)
        self.assertEqual(len(a1), 12400)
        self.assertEqual(a2[:len(a1)], a1)
        self.assertEqual(b1[:live.SHARED], a1[:live.SHARED])
        self.assertNotEqual(b1[live.SHARED], a1[live.SHARED])

    def test_candidate_trace_passes(self):
        captures = live.parse_captures(LOG)
        self.assertEqual(captures["b2"][0]["superseded_evicted"], 1)
        self.assertEqual(captures["references"], [])
        self.assertTrue(all(live.checks(rows(), captures, True).values()))

    def test_lru_eviction_of_the_shared_frontier_fails(self):
        log = list(LOG)
        log[6] = capture(16384, "session", 1, 0)
        result = live.checks(rows(c1=0), live.parse_captures(log), True)
        self.assertFalse(result["a2_evicts_superseded"])
        self.assertFalse(result["c1_shared_8192"])

    def test_bypass_mismatch_fails(self):
        changed = rows()
        changed["b2_ref"]["ids"] = [1, 3]
        self.assertFalse(live.checks(changed, live.parse_captures(LOG), True)["b2_equals_bypass"])

    def test_legacy_expects_a_new_session_miss(self):
        result = live.checks(rows(c1=0), {}, False)
        self.assertTrue(all(result.values()))
        self.assertNotIn("a2_evicts_superseded", result)


if __name__ == "__main__":
    unittest.main()
