import copy
import unittest
from bench.agents import coding, lease, runner, workload
from test_workload import request_ids, text_ids


class CodingTests(unittest.TestCase):
    def turns(self):
        return workload.session(request_ids, text_ids, context_tokens=8192,
                                prefix_tokens=256, suffix_sizes=(64,), output_tokens=200, metrics=False)

    def test_protocol_tee_preserves_stdin_and_stdout(self):
        import json
        import os
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            fake = Path(folder) / 'fake'
            fake.write_text('#!' + sys.executable + '\nimport sys\nfor line in sys.stdin:\n print(line,end="",flush=True)\n')
            fake.chmod(0o700)
            events = Path(folder) / 'events.jsonl'
            body = b'{"event":"ready","context":65536}\n{"event":"token","id":1}\n{"event":"done","cache_bytes":123}\n'
            environment = {**os.environ, 'AGENTS_ENGINE': str(fake), 'AGENTS_EVENTS': str(events)}
            value = subprocess.run([sys.executable, '-m', 'bench.agents.engine_tee'], input=body,
                                   capture_output=True, env=environment, timeout=5)
            self.assertEqual(value.returncode, 0)
            self.assertEqual(value.stdout, body)
            self.assertEqual([json.loads(line)['event'] for line in events.read_text().splitlines()], ['ready', 'done'])

    def test_foreign_listener_fails_preflight(self):
        import socket
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            with self.assertRaises(ValueError):
                lease.require_idle_port(listener.getsockname()[1])

    def test_ready_requires_own_protocol_event(self):
        import tempfile
        import time
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            events, log = root / 'events', root / 'log'
            log.write_text('Crucible is ready at http://127.0.0.1:8790/v1\n')
            response = SimpleNamespace(status=200)
            server = SimpleNamespace(poll=lambda: None)
            with patch.object(lease, 'urlopen') as http:
                http.return_value.__enter__.return_value = response
                with self.assertRaises(TimeoutError):
                    lease.wait_ready(server, time.monotonic() + .01, events, log)
                events.write_text('{"event":"ready","context":65536}\n')
                self.assertIsInstance(lease.wait_ready(server, time.monotonic() + 1, events, log), float)

    def test_done_proof_requires_live_server_and_each_ok_row(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            events = Path(folder) / 'events'
            events.write_text('{"event":"done","request_id":1,"prompt_tokens":42,"completion_tokens":2}\n')
            row = {'status': 'ok', 'usage': {'prompt_tokens': 42, 'completion_tokens': 2},
                   'server_metrics': {'request_id': 1}}
            lease.verify_done(events, [row], alive=True)
            with self.assertRaises(ValueError): lease.verify_done(events, [row], alive=False)
            with self.assertRaises(ValueError): lease.verify_done(events, [row, {**row, 'server_metrics': {'request_id': 2}}], alive=True)

    def test_cache_budget_keeps_nine_gib(self):
        self.assertEqual(lease.cache_budget(40 << 30), 16384)
        self.assertEqual(lease.cache_budget(10 << 30), 1024)
        self.assertEqual(lease.cache_budget(8 << 30), 0)

    def test_source_derived_allocation_bytes(self):
        self.assertEqual(coding.allocated_state_bytes(65536), 2441850880)

    def test_matrix_stops_after_failed_lease(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from bench.agents import matrix
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'matrix'
            args = ['matrix', '--engine', '/binary', '--record', '/record', '--checkpoint', '/checkpoint',
                    '--gpu-run', '/gpu-run', '--root', str(root)]
            with patch('sys.argv', args), patch.object(matrix.subprocess, 'run', return_value=SimpleNamespace(returncode=1)) as run:
                self.assertEqual(matrix.main(), 1)
            self.assertEqual(run.call_count, 1)
            self.assertTrue((root / 'STOP.json').exists())
            self.assertFalse((root / 'PASS.json').exists())
            self.assertEqual(run.call_args.args[0][:9], ['/gpu-run', '-p', '2', '-w', '7200', '--', 'timeout', '-k', '10'])

    def test_reject_mismatched_build_record(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / 'binary'
            binary.write_bytes(b'example')
            with self.assertRaises(ValueError):
                lease.verify_binary(binary, {'binary_sha256': 'invalid'})
            import hashlib
            record = {'binary_sha256': hashlib.sha256(b'example').hexdigest()}
            self.assertEqual(lease.verify_binary(binary, record), record['binary_sha256'])

    def test_policy_preserves_ids_and_original(self):
        source = self.turns()
        original = copy.deepcopy(source)
        turns = coding.configure(source, 'sampled', 3)
        self.assertEqual(source, original)
        self.assertEqual(len(turns), 3)
        for row in turns:
            body = row['body']
            self.assertEqual((body['temperature'], body['top_p'], body['top_k']), (.6, .95, 20))
            self.assertEqual((body['spec'], body['draft_adapt']), (7, 0))
            self.assertEqual(len(request_ids(body)), row['prompt_tokens'])
            self.assertEqual(workload.digest(body), row['request_sha256'])
        runner.validate_plan({'schema': 1, 'concurrency': 1, 'sessions': [turns]})

    def test_invalid_policy_and_short_history(self):
        with self.assertRaises(ValueError): coding.configure(self.turns(), 'invalid', 3)
        with self.assertRaises(ValueError): coding.configure(self.turns()[:1], 'greedy', 3)

    def test_categories_observed_cache_and_large_tool(self):
        row = {'turn': 1, 'suffix_tokens': 512, 'usage': {'prompt_tokens_details': {'cached_tokens': 100}}}
        self.assertEqual(coding.category({**row, 'turn': 0}), 'first_turn')
        self.assertEqual(coding.category(row), 'later_cache_hit')
        self.assertEqual(coding.category({**row, 'turn': 0, 'original_turn': 5, 'cold_segment_start': True}), 'segment_cold_start')
        self.assertEqual(coding.category({**row, 'suffix_tokens': 8192}), 'large_tool_result')
        self.assertEqual(coding.category({**row, 'usage': {}}), 'later_cache_unknown')
        self.assertEqual(coding.category({**row, 'usage': {'prompt_tokens_details': {'cached_tokens': 0}}}), 'later_cache_miss')

    def test_category_summary_and_missing_memory(self):
        def row(turn, suffix, cached):
            return {'session': 0, 'turn': turn, 'suffix_tokens': suffix, 'status': 'ok',
                    'submitted_seconds': turn * 10., 'first_token_seconds': turn * 10. + 1.,
                    'finished_seconds': turn * 10. + 4., 'token_ids': [1, 2, 3],
                    'token_times_seconds': [turn * 10. + 1., turn * 10. + 2., turn * 10. + 3.],
                    'usage': {'completion_tokens': 3, 'prompt_tokens': 10000,
                              'prompt_tokens_details': {'cached_tokens': cached}}}
        result = coding.summarize([row(0, 512, 0), row(1, 512, 100), row(2, 8192, 200)])
        self.assertEqual(result['turn_types']['first_turn']['ttft_seconds']['p50'], 1.)
        self.assertEqual(result['turn_types']['later_cache_hit']['completed_requests'], 1)
        self.assertEqual(result['turn_types']['large_tool_result']['completed_requests'], 1)
        self.assertIsNone(result['memory_peak_bytes'])
        self.assertEqual(result['cache_hit_rate'], 2 / 3)
        self.assertEqual(result['decode_tps_by_session'], [1.])


if __name__ == '__main__': unittest.main()
