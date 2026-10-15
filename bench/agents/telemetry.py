"""Validate the opt-in timing contract of the server without estimating missing data."""
import math
import re

CLOCKS = ('t_submit', 't_admit', 't_first_gpu', 't_prefill_end', 't_first_token', 't_done')
PHASES = ('queue_s', 'restore_s', 'prefill_s', 'between_chunks_s', 'post_prefill_s')
COUNTS = ('restored_tokens', 'prefill_tokens', 'prompt_tokens', 'completion_tokens', 'prefill_chunk_count')
MEMORY = ('pool_bytes', 'live_bytes', 'snapshot_bytes', 'kv_capacity_tokens', 'kv_used_tokens', 'cache_evictions')


def numeric(value):
    return value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0)


def memory(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('Invalid memory observation.')
    result = {key: value.get(key) for key in MEMORY}
    if any(item is not None and (type(item) is not int or item < 0) for item in result.values()):
        raise ValueError('Memory observations must be nonnegative integers or null.')
    capacity, used = result['kv_capacity_tokens'], result['kv_used_tokens']
    if capacity is not None and used is not None and used > capacity:
        raise ValueError('Reported KV use exceeds capacity.')
    return result


def chunks(value):
    if not isinstance(value, list) or len(value) > 262144:
        raise ValueError('Invalid prefill chunk observations.')
    for chunk in value:
        if (not isinstance(chunk, list) or len(chunk) != 3 or
                any(item is None or not numeric(item) for item in chunk[:2]) or chunk[1] < chunk[0] or
                (chunk[2] is not None and (type(chunk[2]) is not int or not 0 <= chunk[2] <= 262144))):
            raise ValueError('Invalid prefill chunk interval.')
    if any(right[0] < left[1] for left, right in zip(value, value[1:])):
        raise ValueError('Overlapping chunks from one request.')
    return [list(chunk) for chunk in value]


def differences(result):
    for field, end, start in [('queue_s', 't_first_gpu', 't_submit'),
                               ('post_prefill_s', 't_first_token', 't_prefill_end')]:
        if all(result[key] is not None for key in (field, end, start)):
            if not math.isclose(result[field], result[end] - result[start], abs_tol=1e-5):
                raise ValueError('Phase duration disagrees with server timestamps.')


def validate(value, usage):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('Invalid server metrics object.')
    result = {key: value.get(key) for key in (*CLOCKS, *PHASES, *COUNTS, 'session_label', 'queued', 'request_id')}
    if any(not numeric(result[key]) for key in (*CLOCKS, *PHASES)):
        raise ValueError('Invalid phase or clock value.')
    known = [result[key] for key in CLOCKS if result[key] is not None]
    if known != sorted(known):
        raise ValueError('Server timestamps are not monotone.')
    if result['session_label'] not in (None, 'miss', 'hit_session', 'hit_shared', 'live'):
        raise ValueError('Unknown server cache label.')
    if result['queued'] is not None and type(result['queued']) is not bool:
        raise ValueError('Queued must be boolean or null.')
    request_id = result['request_id']
    if request_id is not None:
        valid_int = type(request_id) is int and 0 <= request_id < (1 << 64)
        valid_text = isinstance(request_id, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', request_id)
        if not valid_int and not valid_text:
            raise ValueError('Invalid engine request ID.')
    for key in COUNTS:
        if result[key] is not None and (type(result[key]) is not int or result[key] < 0):
            raise ValueError('Invalid server token count.')
    for key in ('prompt_tokens', 'completion_tokens'):
        if result[key] is not None and key in usage and result[key] != usage[key]:
            raise ValueError('Server metrics and usage counts differ.')
    if result['prompt_tokens'] is not None:
        for key in ('restored_tokens', 'prefill_tokens'):
            if result[key] is not None and result[key] > result['prompt_tokens']:
                raise ValueError('Server token count exceeds prompt size.')
    differences(result)
    ttft = result['t_first_token'] - result['t_submit'] if result['t_first_token'] is not None and result['t_submit'] is not None else None
    return {**result, 'memory': memory(value.get('memory')),
            'prefill_chunks': chunks(value.get('prefill_chunks', [])),
            'server_ttft_seconds': ttft, 'restore_is_nested_in_prefill': True}


def interference(session, rows):
    good = [row for row in rows if row.get('status', 'ok') == 'ok']
    if any(row.get('server_metrics') is None or not row.get('server_token_times_seconds') or
           row['server_metrics'].get('prefill_chunk_count') != len(row['server_metrics'].get('prefill_chunks', [])) or
           any(value is None for value in row['server_token_times_seconds']) for row in good):
        return None
    other = [chunk for row in good if row['session'] != session for chunk in row['server_metrics'].get('prefill_chunks', [])]
    return [right - left for row in good if row['session'] == session
            for left, right in zip(row['server_token_times_seconds'], row['server_token_times_seconds'][1:])
            if any(start < right and end > left for start, end, _ in other)]
