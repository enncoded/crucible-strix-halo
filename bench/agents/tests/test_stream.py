import http.client
import io
import json
import unittest
from email.message import Message
from unittest.mock import patch
from bench.agents import stream


def consume(response, **kwargs):
    return stream.consume(response, **{'clock': lambda: 1, 'submitted': 0, 'max_tokens': 2, **kwargs})


def payload(events):
    return io.BytesIO((''.join('data: ' + (json.dumps(event) if isinstance(event, dict) else event) + '\r\n\r\n' for event in events)).encode())


def tokens():
    return [{'choices': [{'delta': {'role': 'assistant', 'content': ''}}]},
            {'choices': [{'delta': {'content': 'a'}, 'token_id': 12}]},
            {'choices': [{'delta': {'content': 'b'}, 'token_id': 13}]},
            {'choices': [{'finish_reason': 'length', 'delta': {}}]},
            {'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 2}}, '[DONE]']


class StreamTests(unittest.TestCase):
    def test_role_chunk_is_not_first_token(self):
        times = iter(range(10, 40))
        row = consume(payload(tokens()), submitted=9, clock=lambda: next(times))
        self.assertEqual(row['token_ids'], [12, 13])
        self.assertEqual(row['first_token_seconds'], row['token_times_seconds'][0])
        self.assertGreater(row['first_token_seconds'], 10)
        self.assertIsNone(row['server_metrics'])

    def test_missing_done_usage_or_ids_rejects(self):
        for events in [tokens()[:-1], tokens()[:-2] + ['[DONE]'],
                       [dict(choices=[{'delta': {'content': 'a'}}]), *tokens()[2:]]]:
            with self.subTest(events=events), self.assertRaises(ValueError):
                consume(payload(events))

    def test_bad_id_nan_duplicate_keys_and_error_reject(self):
        for events, message in [([{'choices': [{'token_id': True}]}], 'token ID'),
                               (['{"choices":[],"choices":[]}'], 'Duplicate'),
                               (['{"usage":{"completion_tokens":NaN}}'], 'Nonfinite'),
                               ([{'error': {'message': 'rejected'}}], 'failed stream')]:
            with self.subTest(events=events), self.assertRaisesRegex(ValueError, message):
                consume(payload(events))

    def test_bounds_and_overproduction(self):
        with self.assertRaisesRegex(ValueError, 'excess output'):
            consume(payload(tokens()), max_tokens=1)
        with self.assertRaisesRegex(ValueError, 'byte limit'):
            consume(io.BytesIO(b'data: ' + b'x' * 100 + b'\n'), line_limit=32)

    def test_server_contract_is_in_usage_and_root_token_chunk(self):
        events = tokens()
        events[1] = {**events[1], 'crucible_t': 1.2}
        events[2] = {**events[2], 'crucible_t': 1.3}
        events[4] = {'choices': [], 'usage': {**events[4]['usage'], 'crucible_metrics': {'session_label': 'miss'}}}
        row = consume(payload(events))
        self.assertEqual(row['server_token_times_seconds'], [1.2, 1.3])
        self.assertEqual(row['server_metrics'], {'session_label': 'miss'})

    def test_prefix_and_whitespace_chunks_keep_token_timestamps(self):
        events = tokens(); events[1] = {'choices': [{'delta': {'content': ''}, 'token_id': 12}]}
        self.assertEqual(len(consume(payload(events))['token_times_seconds']), 2)


if __name__ == '__main__': unittest.main()


class TruncatedResponse:
    """An SSE response whose connection drops after one event."""
    def __init__(self):
        self.headers = Message()
        self.headers['Content-Type'] = 'text/event-stream'
        self.lines = iter([b'data: ' + json.dumps(tokens()[0]).encode() + b'\r\n', b'\r\n'])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def readline(self, limit=-1):
        for line in self.lines:
            return line
        raise http.client.IncompleteRead(b'')


class TruncatedStreamTests(unittest.TestCase):
    def test_truncated_stream_returns_an_error_row(self):
        body = dict(stream=True, return_token_ids=True, max_tokens=2)
        with patch.object(stream, 'urlopen', return_value=TruncatedResponse()):
            row = stream.request('http://127.0.0.1:1/v1/chat/completions', body, session=3, turn=4, clock=lambda: 5.0)
        self.assertEqual(row['status'], 'error')
        self.assertEqual(row['error_type'], 'IncompleteRead')
        self.assertEqual((row['session'], row['turn']), (3, 4))
