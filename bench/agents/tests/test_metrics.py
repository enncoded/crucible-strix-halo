import unittest
from bench.agents import metrics


def turn(session=0, start=0, first=1, times=(1, 2, 3)):
    return {'status': 'ok', 'session': session, 'turn': 0,
            'submitted_seconds': start, 'first_token_seconds': first,
            'finished_seconds': times[-1] + .1, 'token_times_seconds': list(times),
            'token_ids': [12] * len(times), 'usage': {'prompt_tokens': 100, 'completion_tokens': len(times)},
            'server_metrics': None}


class MetricsTests(unittest.TestCase):
    def test_nearest_rank(self):
        self.assertEqual(metrics.percentile([3, 1, 2], .5), 2)
        self.assertEqual(metrics.percentile([3, 1, 2], .95), 3)
        self.assertIsNone(metrics.percentile([], .5))

    def test_per_turn_and_aggregate_denominators(self):
        rows = [turn(), turn(1, 0, 2, (2, 3, 4))]
        result = metrics.summarize(rows, decode_ceiling=2)
        self.assertEqual(result['sessions'][0]['decode_tps'], 1)
        self.assertEqual(result['sessions'][0]['decode_ceiling_pct'], 50)
        self.assertAlmostEqual(result['aggregate_generation_tps'], 6 / 4.1)
        self.assertEqual(result['ttft_seconds']['p95'], 2)
        self.assertIsNone(result['kv_pool_capacity_tokens'])
        self.assertIsNone(result['sessions'][0]['cache_hit_rate'])

    def test_rejections_stay_in_report(self):
        result = metrics.summarize([turn(), {'status': 'rejected', 'session': 1, 'turn': 0, 'http_status': 429}])
        self.assertEqual(result['rejected_requests'], 1)
        self.assertEqual(result['sessions'][1]['completed_turns'], 0)

    def test_invalid_times_and_counts_fail(self):
        for row in [dict(turn(), token_times_seconds=[1, .5, 2]),
                    dict(turn(), first_token_seconds=float('nan')),
                    dict(turn(), usage={'completion_tokens': 4})]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                metrics.summarize([row])

    def test_no_fabricated_ceiling_or_prefill_interference(self):
        result = metrics.summarize([turn()])
        self.assertIsNone(result['sessions'][0]['decode_ceiling_pct'])
        self.assertIsNone(result['sessions'][0]['inter_token_during_other_prefill_seconds'])

    def test_phase_and_memory_observations(self):
        row = {**turn(), 'server_token_times_seconds': [2, 3, 4], 'server_metrics': {
            't_submit': 0, 't_first_gpu': 1, 't_first_token': 2, 't_done': 4.1,
            'queue_s': 1, 'queued': True, 'restored_tokens': 90,
            'memory': {'kv_capacity_tokens': 200, 'kv_used_tokens': 100, 'pool_bytes': 4096}}}
        result = metrics.summarize([row])
        self.assertEqual(result['kv_pool_capacity_tokens'], 200)
        self.assertEqual(result['queued_requests'], 1)
        self.assertEqual(result['sessions'][0]['restored_tokens'], 90)
        self.assertEqual(result['phases']['queue_s']['p50'], 1)
        self.assertIsNone(result['phases']['restore_s']['p50'])

    def test_no_mutation(self):
        import copy
        rows = [turn()]; before = copy.deepcopy(rows)
        metrics.summarize(rows)
        self.assertEqual(rows, before)


if __name__ == '__main__': unittest.main()
