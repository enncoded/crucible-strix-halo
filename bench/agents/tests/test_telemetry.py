import unittest
from bench.agents import telemetry


def observation():
    return {'request_id': 'req-1', 'session_label': 'hit_session', 'queued': True,
            't_submit': 0, 't_admit': 1, 't_first_gpu': 1, 't_prefill_end': 4,
            't_first_token': 4.2, 't_done': 6, 'queue_s': 1, 'restore_s': .1,
            'prefill_s': 2, 'between_chunks_s': 1, 'post_prefill_s': .2,
            'restored_tokens': 90, 'prefill_tokens': 10, 'prompt_tokens': 100, 'completion_tokens': 3,
            'prefill_chunks': [[1, 2, 5], [3, 4, 5]],
            'memory': {'pool_bytes': 4096, 'live_bytes': 2048, 'snapshot_bytes': 1024,
                       'kv_capacity_tokens': 200, 'kv_used_tokens': 100, 'cache_evictions': 0}}


class TelemetryTests(unittest.TestCase):
    def test_server_split_and_nested_restore(self):
        result = telemetry.validate(observation(), {'prompt_tokens': 100, 'completion_tokens': 3})
        self.assertAlmostEqual(result['server_ttft_seconds'], 4.2)
        self.assertTrue(result['restore_is_nested_in_prefill'])
        self.assertEqual(result['memory']['kv_capacity_tokens'], 200)

    def test_missing_data_remains_unknown(self):
        self.assertIsNone(telemetry.validate(None, {}))
        result = telemetry.validate({'t_submit': 1, 't_first_gpu': 2, 'queue_s': 1}, {})
        self.assertIsNone(result['prefill_s'])
        self.assertIsNone(result['memory'])

    def test_bad_times_counts_phases_and_capacity_reject(self):
        for changes in [{'t_first_token': 3}, {'queue_s': 2}, {'restore_s': -1},
                        {'completion_tokens': 4}, {'queued': 1}, {'session_label': 'guess'},
                        {'prefill_chunks': [[2, 1, 5]]},
                        {'memory': {'kv_capacity_tokens': 2, 'kv_used_tokens': 3}},
                        {'prefill_s': float('nan')}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                telemetry.validate({**observation(), **changes}, {'prompt_tokens': 100, 'completion_tokens': 3})

    def test_truncated_chunk_data_cannot_claim_complete_interference(self):
        row = {'session': 0, 'server_token_times_seconds': [1, 2],
               'server_metrics': {'prefill_chunk_count': 2, 'prefill_chunks': [[0, 1, 8]]}}
        self.assertIsNone(telemetry.interference(0, [row]))

    def test_overlap_uses_server_clock_and_other_sessions_only(self):
        rows = [{'session': 0, 'server_token_times_seconds': [1, 2, 3],
                 'server_metrics': {'prefill_chunk_count': 1, 'prefill_chunks': [[1, 3, 10]]}},
                {'session': 1, 'server_token_times_seconds': [4, 5],
                 'server_metrics': {'prefill_chunk_count': 1, 'prefill_chunks': [[1.5, 2.5, 10]]}}]
        self.assertEqual(telemetry.interference(0, rows), [1, 1])
        self.assertEqual(telemetry.interference(1, rows), [])
        self.assertIsNone(telemetry.interference(0, [{'session': 0, 'server_metrics': None}]))


if __name__ == '__main__': unittest.main()
