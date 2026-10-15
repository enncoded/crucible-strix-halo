"""Measured streaming latency. Missing server observations remain null."""
import math
from . import telemetry


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def percentile(values, probability):
    if not 0 < probability <= 1 or any(not finite(value) for value in values):
        raise ValueError('Percentiles need finite nonnegative values and a probability in (0,1].')
    return sorted(values)[math.ceil(len(values) * probability) - 1] if values else None


def distribution(values):
    return {'samples': len(values), 'p50': percentile(values, .5),
            'p95': percentile(values, .95), 'p99': percentile(values, .99)}


def validate(row):
    if row.get('status') not in ('ok', 'rejected', 'error'):
        raise ValueError('Unknown request status.')
    if any(type(row.get(key)) is not int or row[key] < 0 for key in ('session', 'turn')):
        raise ValueError('Session and turn IDs must be nonnegative integers.')
    if row['status'] != 'ok':
        return
    times, ids = row['token_times_seconds'], row['token_ids']
    start, first, end = (row[key] for key in ('submitted_seconds', 'first_token_seconds', 'finished_seconds'))
    if (not times or len(ids) != len(times) or any(not finite(value) for value in [start, first, end, *times]) or
            times != sorted(times) or not start <= first == times[0] <= times[-1] <= end):
        raise ValueError('Invalid token times or request interval.')
    if any(type(value) is not int or not 0 <= value < 248320 for value in ids):
        raise ValueError('Invalid returned token ID.')
    usage = row['usage']
    if type(usage.get('completion_tokens')) is not int or usage['completion_tokens'] != len(ids):
        raise ValueError('Returned token IDs and usage differ.')
    if type(usage.get('prompt_tokens')) is not int or usage['prompt_tokens'] <= 0:
        raise ValueError('Invalid prompt usage.')
    server_times = row.get('server_token_times_seconds', [])
    if server_times:
        known = [value for value in server_times if value is not None]
        if len(server_times) != len(ids) or any(not finite(value) for value in known) or known != sorted(known):
            raise ValueError('Invalid server token times.')
    telemetry.validate(row.get('server_metrics'), usage)
    cached = usage.get('prompt_tokens_details', {}).get('cached_tokens')
    if cached is not None and (type(cached) is not int or not 0 <= cached <= usage['prompt_tokens']):
        raise ValueError('Invalid cached token count.')


def decode_rate(rows):
    tokens = sum(max(0, len(row['token_ids']) - 1) for row in rows)
    seconds = sum(row['token_times_seconds'][-1] - row['first_token_seconds'] for row in rows)
    return tokens / seconds if seconds > 0 else None


def session_summary(session, rows, ceiling):
    good = [row for row in rows if row['status'] == 'ok']
    rate = decode_rate(good)
    cached = [row['usage'].get('prompt_tokens_details', {}).get('cached_tokens') for row in good]
    known = [count for count in cached if count is not None]
    gaps = [right - left for row in good for left, right in zip(row['token_times_seconds'], row['token_times_seconds'][1:])]
    return {'session': session, 'completed_turns': len(good),
            'ttft_seconds': distribution([row['first_token_seconds'] - row['submitted_seconds'] for row in good]),
            'decode_tps': rate, 'decode_ceiling_tps': ceiling,
            'decode_ceiling_pct': 100 * rate / ceiling if rate is not None and ceiling is not None else None,
            'inter_token_seconds': distribution(gaps),
            'inter_token_during_other_prefill_seconds': None,
            'cached_tokens': sum(known) if known else None,
            'cache_hit_rate': sum(count > 0 for count in known) / len(known) if known else None,
            'cache_observed_turns': len(known),
            'restored_tokens': observed_sum(good, 'restored_tokens')}


def observed_sum(rows, key):
    values = [row['server_metrics'][key] for row in rows if row.get('server_metrics') and row['server_metrics'][key] is not None]
    return sum(values) if values else None


def observations(rows, sessions):
    good = [row for row in rows if row['status'] == 'ok']
    known = [row for row in good if row.get('server_metrics')]
    memory = [row for row in known if row['server_metrics']['memory'] is not None]
    latest = max(memory, key=lambda row: row['finished_seconds'])['server_metrics']['memory'] if memory else None
    queued = [row['server_metrics']['queued'] for row in known if row['server_metrics']['queued'] is not None]
    def gaps(session):
        values = telemetry.interference(session, good)
        return distribution(values) if values is not None else None
    return {'sessions': [{**session, 'inter_token_during_other_prefill_seconds': gaps(session['session'])} for session in sessions],
            'phases': {key: distribution([row['server_metrics'][key] for row in known if row['server_metrics'][key] is not None])
                       for key in telemetry.PHASES},
            'queued_requests': sum(queued) if queued else None, 'queue_observed_requests': len(queued),
            'kv_pool_capacity_tokens': latest['kv_capacity_tokens'] if latest else None,
            'kv_pool_used_tokens': latest['kv_used_tokens'] if latest else None, 'last_observed_memory': latest,
            'restore_is_nested_in_prefill': True}


def summarize(rows, *, decode_ceiling=None):
    if decode_ceiling is not None and (not finite(decode_ceiling) or decode_ceiling == 0):
        raise ValueError('The decode ceiling must be positive and finite.')
    for row in rows:
        validate(row)
    if len({(row['session'], row['turn']) for row in rows}) != len(rows):
        raise ValueError('Duplicate session/turn measurement.')
    rows = [{**row, 'server_metrics': telemetry.validate(row.get('server_metrics'), row.get('usage', {}))} for row in rows]
    good = [row for row in rows if row['status'] == 'ok']
    span = max(row['finished_seconds'] for row in good) - min(row['submitted_seconds'] for row in good) if good else 0
    sessions = [session_summary(session, [row for row in rows if row['session'] == session], decode_ceiling)
                for session in sorted({row['session'] for row in rows})]
    return {'ttft_seconds': distribution([row['first_token_seconds'] - row['submitted_seconds'] for row in good]),
            'sessions': sessions, 'completed_requests': len(good),
            'rejected_requests': sum(row['status'] == 'rejected' for row in rows),
            'error_requests': sum(row['status'] == 'error' for row in rows),
            'aggregate_generation_tps': sum(len(row['token_ids']) for row in good) / span if span > 0 else None,
            'aggregate_wall_seconds': span, 'queued_requests': None,
            'kv_pool_capacity_tokens': None, 'kv_pool_used_tokens': None, 'memory_in_use_bytes': None,
            'percentile_method': 'nearest rank',
            'decode_denominator': 'Sum of first-to-last token intervals per session; numerator excludes each first token.',
            'aggregate_denominator': 'First submission to last completion, including queue, prefill and decode.',
            **observations(rows, sessions)}
