import unittest
from bench.agents import assemble


class AssembleTests(unittest.TestCase):
    def test_aggregate_excludes_inter_lease_wait_and_preserves_turn_ids(self):
        def packet(start, original):
            row = {'status': 'ok', 'session': 0, 'turn': 0, 'original_turn': original,
                   'cold_segment_start': True, 'suffix_tokens': 512,
                   'submitted_seconds': start, 'first_token_seconds': start + 1,
                   'finished_seconds': start + 3, 'token_times_seconds': [start + 1, start + 2],
                   'token_ids': [1, 2], 'usage': {'prompt_tokens': 100, 'completion_tokens': 2,
                   'prompt_tokens_details': {'cached_tokens': 0}}}
            report = {'rows': [row], 'status': 'PASS', 'summary': {'aggregate_wall_seconds': 3},
                      'binary_sha256': 'abc'}
            receipt = {'status': 'PASS', 'binary_sha256': 'abc', 'inputs_unchanged': True,
                       'build_record_sha256': 'record', 'cache_budget_mib': 100, 'lease_seconds': 10, 'resources': {}}
            return report, receipt
        result = assemble.aggregate([packet(0., 0), packet(100., 5)])
        self.assertEqual(result['aggregate_generation_tps'], 4 / 6)
        self.assertEqual(result['turn_types']['first_turn']['completed_requests'], 1)
        self.assertEqual(result['turn_types']['segment_cold_start']['completed_requests'], 1)
        self.assertEqual(result['completed_requests'], 2)
        self.assertEqual(result['lease_seconds'], [10, 10])

    def test_collect_rejects_missing_segment(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            segment = root / 't5-batch-k4-greedy-s0'
            segment.mkdir()
            (segment / 'report.json').write_text(json.dumps({'status': 'PASS', 'rows': [],
                'summary': {'aggregate_wall_seconds': 0}, 'binary_sha256': 'abc'}))
            (segment / 'lease.json').write_text(json.dumps({'status': 'PASS', 'binary_sha256': 'abc',
                'build_record_sha256': 'record', 'inputs_unchanged': True, 'lease_seconds': 10, 'cache_budget_mib': 100}))
            with self.assertRaisesRegex(ValueError, 'segment'):
                assemble.collect(root)

    def test_collect_rejects_missing_receipt(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            segment = root / 't5-batch-k1-greedy'
            segment.mkdir()
            (segment / 'report.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'receipt'):
                assemble.collect(root)

    def test_aggregate_rejects_mixed_binaries_and_records(self):
        def packet(binary, record):
            report = {'status': 'PASS', 'rows': [], 'summary': {'aggregate_wall_seconds': 0}, 'binary_sha256': binary}
            receipt = {'status': 'PASS', 'inputs_unchanged': True, 'binary_sha256': binary,
                       'build_record_sha256': record, 'cache_budget_mib': 100, 'lease_seconds': 1}
            return report, receipt
        with self.assertRaisesRegex(ValueError, 'binary'):
            assemble.aggregate([packet('a', 'one'), packet('b', 'two')])
        with self.assertRaisesRegex(ValueError, 'record'):
            assemble.aggregate([packet('a', 'one'), packet('a', 'two')])

    def test_aggregate_rejects_failed_receipt(self):
        with self.assertRaises(ValueError):
            assemble.aggregate([({'status': 'PASS', 'binary_sha256': 'abc'}, {'status': 'FAIL'})])


if __name__ == '__main__': unittest.main()
