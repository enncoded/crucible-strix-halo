"""Frozen agent-shaped workload and observed per-turn categories."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from . import metrics, runner, workload

SUFFIXES = (512, 1024, 2048, 512, 8192)
TURN_TYPES = ('first_turn', 'later_cache_hit', 'large_tool_result', 'later_cache_miss', 'later_cache_unknown', 'segment_cold_start')


def configure(turns, policy, count):
    if policy not in ('greedy', 'sampled') or type(count) is not int or not 1 <= count <= len(turns):
        raise ValueError('Use greedy or sampled and a turn count that fits the history.')
    sampling = {'temperature': .6, 'top_p': .95, 'top_k': 20} if policy == 'sampled' else {'temperature': 0}
    def update(row):
        body = {**row['body'], **sampling, 'spec': 7, 'draft_adapt': 0,
                'draft_head': '64k', 'draft_q': .3, 'mtp_prefix': 256}
        return {**row, 'body': body, 'request_sha256': workload.digest(body), 'sampling_policy': policy}
    return [update(row) for row in turns[:count]]


def prepare(checkpoint, count, policy, turns=15):
    from server.api import validate_request
    from server.tokenization import ModelTokenizer
    tokenizer = ModelTokenizer(checkpoint)
    def encode(body):
        return tokenizer.encode(validate_request(body, 'chat', 'crucible'))
    def text(value):
        return tokenizer.tokenizer.encode(value, add_special_tokens=False).ids
    sessions = [configure(workload.session(encode, text, session=n, suffix_sizes=SUFFIXES,
                                          output_tokens=200, context_tokens=65536), policy, turns)
                for n in range(count)]
    paths = [Path(checkpoint) / name for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')]
    first = sessions[0][0]
    shared = tokenizer.shared_prefix_tokens(validate_request(first['body'], 'chat', 'crucible'), encode(first['body']))
    result = {'schema': 1, 'concurrency': count, 'sessions': sessions, 'context_limit_tokens': 65536,
              'turns_per_session': turns, 'output_tokens': 200, 'teacher_forced': True,
              'shared_prefix_tokens': shared, 'policy': policy,
              'tokenizer_hashes': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.exists()},
              'tool_hashes': runner.tool_hashes(), 'coding_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    runner.validate_plan(result)
    return result


def category(row):
    if row.get('cold_segment_start') and row.get('original_turn', 0) > 0:
        return 'segment_cold_start'
    if row.get('original_turn', row['turn']) == 0:
        return 'first_turn'
    if row['suffix_tokens'] >= 4096:
        return 'large_tool_result'
    cached = row.get('usage', {}).get('prompt_tokens_details', {}).get('cached_tokens')
    return 'later_cache_unknown' if cached is None else 'later_cache_hit' if cached > 0 else 'later_cache_miss'


def summarize(rows):
    good = [row for row in rows if row['status'] == 'ok']
    groups = {key: metrics.summarize([row for row in good if category(row) == key]) for key in TURN_TYPES}
    cached = [row['usage'].get('prompt_tokens_details', {}).get('cached_tokens') for row in good]
    known = [value for value in cached if value is not None]
    memory = [row['server_metrics']['memory'] for row in good
              if row.get('server_metrics') and row['server_metrics'].get('memory')]
    live = [value['live_bytes'] for value in memory if value.get('live_bytes') is not None]
    snapshots = [value['snapshot_bytes'] for value in memory if value.get('snapshot_bytes') is not None]
    rates = [value['decode_tps'] for value in metrics.summarize(rows)['sessions']]
    return {'turn_types': groups, 'cache_hit_rate': sum(value > 0 for value in known) / len(known) if known else None,
            'cache_observed_requests': len(known), 'decode_tps_by_session': rates,
            'memory_peak_bytes': max(live) if live else None,
            'snapshot_peak_bytes': max(snapshots) if snapshots else None}


def allocated_state_bytes(context):
    # Mirror the pinned full-storage SequenceState::Bytes formula, not a measured dynamic pool.
    geometry = Path('engine/model/flashnext/geometry.h').read_text()
    operations = Path('engine/layers/ops.h').read_text()
    rows = int(re.search(r'kMaxRows = (\d+);', geometry).group(1))
    pending = int(re.search(r'kPend = (\d+);', operations).group(1))
    fixed = 36 * (48 * 128 * 128 + 16 * 10240 + 48 * rows * pending) * 4 + 16 * 10240 * 4
    qsa = 13 * (context * (1024 + 1024 + 512) + ((context + 127) // 128) * 4096 * 4)
    return fixed + qsa


def resources(folder):
    events_path = folder / 'engine-events.jsonl'
    events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
    ready = next((row for row in events if row.get('event') == 'ready'), None)
    contexts = ready.get('slot_contexts', [ready['context']]) if ready else None
    caches = [row['cache_bytes'] for row in events if type(row.get('cache_bytes')) is int]
    trace_path = folder / 'trace.jsonl'
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()] if trace_path.exists() else []
    available = [row['mem_available_bytes'] for row in trace if isinstance(row, dict) and 'mem_available_bytes' in row]
    return {'slot_contexts': contexts, 'configured_capacity_tokens': sum(contexts) if contexts else None,
            'source_derived_allocated_state_bytes': sum(allocated_state_bytes(c) for c in contexts) if contexts else None,
            'snapshot_cache_peak_bytes': max(caches) if caches else None,
            'mem_available_floor_bytes': min(available) if available else None,
            'mem_available_samples': len(available), 'ready_event': ready}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--sessions', type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument('--policy', choices=('greedy', 'sampled'), required=True)
    parser.add_argument('--turns', type=int, default=15)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runner.write_new(args.output, prepare(args.checkpoint, args.sessions, args.policy, args.turns))


if __name__ == '__main__': main()
