import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from bench.agents import runner, workload
from test_workload import request_ids, text_ids


class MockServer(ThreadingHTTPServer):
    request_queue_size = 64


class MockEngineHandler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        with self.server.guard:
            self.server.active += 1
            self.server.peak = max(self.server.peak, self.server.active)
            first = body['seed'] not in self.server.seen
            self.server.seen = {*self.server.seen, body['seed']}
        try:
            if first and self.server.barrier is not None:
                self.server.barrier.wait(timeout=10)
            if self.server.reject:
                self.send_error(429); return
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
            def emit(event):
                self.wfile.write(('data: ' + (event if isinstance(event, str) else json.dumps(event)) + '\n\n').encode()); self.wfile.flush()
            emit({'choices': [{'delta': {'role': 'assistant', 'content': ''}}]})
            time.sleep(.02)
            for index in range(body['max_tokens']):
                emit({'choices': [{'delta': {'content': 'x'}, 'token_id': 12 + index % 10}]})
            emit({'choices': [], 'usage': {'prompt_tokens': len(request_ids(body)), 'completion_tokens': body['max_tokens']}})
            emit('[DONE]')
        finally:
            with self.server.guard: self.server.active -= 1


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.server = MockServer(('127.0.0.1', 0), MockEngineHandler)
        self.server.guard = threading.Lock(); self.server.active = 0; self.server.peak = 0; self.server.reject = False
        self.server.seen = set(); self.server.barrier = None
        self.thread = threading.Thread(target=self.server.serve_forever); self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/v1/chat/completions'

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()

    def plan(self, count):
        sessions = [workload.session(request_ids, text_ids, session=n, context_tokens=2048,
                    prefix_tokens=256, suffix_sizes=(64,), output_tokens=200, metrics=False)[:2] for n in range(count)]
        return {'schema': 1, 'sessions': sessions, 'concurrency': count}

    def test_all_concurrency_levels_real_http(self):
        for count in (1, 2, 4, 8):
            with self.subTest(count=count):
                self.server.peak = 0; self.server.seen = set(); self.server.barrier = threading.Barrier(count)
                result = runner.run(self.plan(count), self.url, label='cpu-mock')
                self.assertEqual(result['summary']['completed_requests'], count * 2)
                self.assertGreaterEqual(self.server.peak, count)
                self.assertEqual(len(result['rows']), count * 2)
                self.assertTrue(all(len(row['token_ids']) == 200 for row in result['rows']))
                self.assertIsNone(result['summary']['kv_pool_capacity_tokens'])

    def test_rejections_and_incomplete_plan_fail_closed(self):
        self.server.reject = True
        result = runner.run(self.plan(2), self.url, label='cpu-mock')
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['summary']['rejected_requests'], 2)
        with self.assertRaises(ValueError):
            runner.run({'schema': 1, 'sessions': [], 'concurrency': 2}, self.url, label='cpu-mock')

    def test_mutated_request_rejected_before_http(self):
        plan = self.plan(1); plan['sessions'][0][0]['body']['seed'] += 1
        with self.assertRaises(ValueError): runner.run(plan, self.url, label='cpu-mock')
        self.assertEqual(self.server.peak, 0)


if __name__ == '__main__': unittest.main()
