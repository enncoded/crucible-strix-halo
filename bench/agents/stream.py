"""Bounded OpenAI SSE reader. Count token IDs, not display chunks."""
import http.client
import json
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LINE_LIMIT = 1024 * 1024
BYTE_LIMIT = 16 * 1024 * 1024


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key in stream.')
        result = {**result, key: value}
    return result


def invalid_constant(value):
    raise ValueError('Nonfinite JSON number in stream.')


def events(response, *, line_limit, deadline, clock):
    data, count = [], 0
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            raise ValueError('The streaming request exceeded its time limit.')
        sock = getattr(getattr(getattr(response, 'fp', None), 'raw', None), '_sock', None)
        if sock is not None:
            sock.settimeout(remaining)
        line = response.readline(line_limit + 1)
        count += len(line)
        if len(line) > line_limit or count > BYTE_LIMIT:
            raise ValueError('The stream exceeded its byte limit.')
        if not line:
            raise ValueError('The stream ended without [DONE].')
        text = line.decode('utf-8').rstrip('\r\n')
        if text.startswith('data:'):
            data = [*data, text[5:].lstrip(' ')]
        elif not text and data:
            yield '\n'.join(data)
            data = []


def accept_event(event, ids, times, texts, server_times, timestamp, max_tokens):
    if not isinstance(event, dict) or 'error' in event:
        raise ValueError('The server returned an invalid or failed stream event.')
    choices = event.get('choices', [])
    if not isinstance(choices, list) or len(choices) > 1:
        raise ValueError('The stream must contain at most one completion choice.')
    for choice in choices:
        if not isinstance(choice, dict):
            raise ValueError('Invalid completion choice.')
        content = choice.get('delta', {}).get('content', '')
        if content is not None and not isinstance(content, str):
            raise ValueError('Invalid streamed text.')
        token = choice.get('token_id')
        if token is None and content:
            raise ValueError('Token IDs are required for measured output.')
        if token is not None:
            if type(token) is not int or not 0 <= token < 248320 or len(ids) >= max_tokens:
                raise ValueError('Invalid token ID or excess output.')
            observed = event.get('crucible_t')
            if observed is not None and (type(observed) not in (int, float) or observed < 0):
                raise ValueError('Invalid server token timestamp.')
            ids = [*ids, token]; times = [*times, timestamp]; server_times = [*server_times, observed]
        if content:
            texts = [*texts, content]
    usage = event.get('usage')
    if usage is not None and not isinstance(usage, dict):
        raise ValueError('Invalid usage object.')
    return ids, times, texts, server_times, usage, usage.get('crucible_metrics') if usage else None


def consume(response, *, submitted, max_tokens, clock=time.monotonic, timeout=300, line_limit=LINE_LIMIT):
    if type(max_tokens) is not int or not 1 <= max_tokens <= 800 or not 0 < timeout <= 900:
        raise ValueError('Output must fit 1..800 tokens and timeout must fit (0,900] seconds.')
    ids, times, texts, server_times, usage, server = [], [], [], [], None, None
    for raw in events(response, line_limit=line_limit, deadline=submitted + timeout, clock=clock):
        timestamp = clock()
        if raw == '[DONE]':
            if not ids or not isinstance(usage, dict) or usage.get('completion_tokens') != len(ids):
                raise ValueError('Complete token IDs and matching usage are required.')
            return {'status': 'ok', 'submitted_seconds': submitted, 'first_token_seconds': times[0],
                    'finished_seconds': timestamp, 'token_ids': ids, 'token_times_seconds': times,
                    'output_text': ''.join(texts), 'usage': usage, 'server_metrics': server,
                    'server_token_times_seconds': server_times}
        event = json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)
        ids, times, texts, server_times, observed_usage, observed_server = accept_event(
            event, ids, times, texts, server_times, timestamp, max_tokens)
        if observed_usage is not None:
            if usage is not None:
                raise ValueError('Duplicate usage in stream.')
            usage = observed_usage
        if observed_server is not None:
            if server is not None or not isinstance(observed_server, dict):
                raise ValueError('Invalid or duplicate server metrics.')
            server = observed_server


def request(url, body, *, session, turn, timeout=300, clock=time.monotonic):
    data = json.dumps(body, allow_nan=False).encode('utf-8')
    if len(data) > BYTE_LIMIT or body.get('stream') is not True or body.get('return_token_ids') is not True:
        raise ValueError('Use a bounded streaming request with token IDs.')
    started = clock()
    identity = {'session': session, 'turn': turn}
    req = Request(url, data=data, headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
    try:
        with urlopen(req, timeout=timeout) as response:
            if response.headers.get_content_type() != 'text/event-stream':
                raise ValueError('The server did not return an SSE stream.')
            return {**identity, **consume(response, submitted=started, max_tokens=body['max_tokens'], clock=clock, timeout=timeout)}
    except HTTPError as error:
        error.close()
        return {**identity, 'status': 'rejected' if error.code in (400, 409, 429, 503) else 'error',
                'http_status': error.code, 'submitted_seconds': started, 'finished_seconds': clock()}
    except (ValueError, OSError, URLError, socket.timeout, http.client.HTTPException) as error:
        # A truncated stream raises HTTPException (for example IncompleteRead); record it, do not stop the run.
        # Server text can contain credentials. Keep only the exception class.
        return {**identity, 'status': 'error', 'error_type': type(error).__name__,
                'submitted_seconds': started, 'finished_seconds': clock()}
