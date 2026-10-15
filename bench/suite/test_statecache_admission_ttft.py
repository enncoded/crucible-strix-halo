import unittest
from tools.statecache_admission_ttft import verify, summaries


class AdmissionTimingTests(unittest.TestCase):
    def rows(self):
        return [dict(case="4k", spec=0, repeat=0, phase=phase, output_ids=[1] * 32,
                     ttft_seconds=2., total_seconds=3., cached_tokens=0, prefill_tokens=4088,
                     cache_bytes=0, cache_saves=0, cache_saved_bytes=0, input_tokens=4088, sampling={}) for phase in ("before", "miss", "retry", "after")]

    def test_short_default_saves_nothing_and_reports_total(self):
        rows = self.rows()
        verify(rows, {(r["case"], r["spec"], r["repeat"], r["phase"]) for r in rows})
        result = summaries(rows, 2310)[0]
        self.assertEqual(result["miss_total_overhead_percent"], 0)
        self.assertEqual(result["retry_hit_rate"], 0)
        for key in ("cached_tokens", "cache_bytes", "cache_saves", "cache_saved_bytes"):
            changed = [r | {key: 1} if r["phase"] == "miss" else r for r in rows]
            with self.assertRaises(RuntimeError):
                verify(changed, {(r["case"], r["spec"], r["repeat"], r["phase"]) for r in rows})

    def test_cached_bypass_and_refresh_rows_fail(self):
        rows = self.rows()
        for phase in ("before", "miss", "after"):
            bad = [r | dict(case="exact4k", cached_tokens=4088 if r["phase"] == phase or r["phase"] == "retry" else 0,
                            prefill_tokens=0 if r["phase"] == phase or r["phase"] == "retry" else 4088,
                            cache_saves=int(r["phase"] == "miss"), cache_saved_bytes=100 if r["phase"] == "miss" else 0)
                   for r in rows]
            expected = {(r["case"], r["spec"], r["repeat"], r["phase"]) for r in bad}
            with self.assertRaises(RuntimeError): verify(bad, expected)

    def test_all_ids_and_complete_turn_coverage_are_binding(self):
        rows = self.rows(); expected = {(r["case"], r["spec"], r["repeat"], r["phase"]) for r in rows}
        for bad in (rows[:-1], rows + rows[:1],
                    [r | dict(output_ids=[2] * 32) if r["phase"] == "retry" else r for r in rows]):
            with self.assertRaises(RuntimeError): verify(bad, expected)

    def test_long_saves_only_the_newest_frontier(self):
        rows = []
        for turn in range(3):
            for phase in ("bypass", "auto"):
                row = self.rows()[0] | dict(case="30k", spec=2, phase=phase + str(turn))
                if phase == "auto":
                    row |= dict(cached_tokens=(0, 24576, 28672)[turn],
                                cache_saves=int(turn < 2), cache_saved_bytes=100 if turn < 2 else 0)
                rows.append(row)
        expected = {(r["case"], r["spec"], r["repeat"], r["phase"]) for r in rows}
        verify(rows, expected)
        for count in (0, 2):
            bad = [r | dict(cache_saves=count) if r["phase"] == "auto0" else r for r in rows]
            with self.assertRaises(RuntimeError): verify(bad, expected)

    def test_long_later_turn_and_exact_opt_in_must_hit(self):
        for case in ("16k", "30k", "exact4k"):
            rows = [r | dict(case=case) for r in self.rows()]
            if case != "exact4k":
                rows = [r | dict(phase=phase) for r, phase in zip(rows, ("bypass0", "auto0", "bypass1", "auto1"))]
            expected = {(r["case"], r["spec"], r["repeat"], r["phase"]) for r in rows}
            with self.assertRaises(RuntimeError): verify(rows, expected)


if __name__ == "__main__":
    unittest.main()
