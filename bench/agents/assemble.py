"""Combine completed short leases without counting the queue wait as generation time."""
import argparse
import json
from pathlib import Path
import re
from . import coding, metrics, runner


def aggregate(packets):
    if not packets or any(report['status'] != 'PASS' or receipt['status'] != 'PASS' or
                          not receipt.get('inputs_unchanged') or
                          report['binary_sha256'] != receipt.get('binary_sha256') for report, receipt in packets):
        raise ValueError('Aggregate only complete measurements with valid receipts.')
    if len({report['binary_sha256'] for report, _ in packets}) != 1:
        raise ValueError('A group cannot mix binary hashes.')
    records = {receipt.get('build_record_sha256') for _, receipt in packets}
    if len(records) != 1 or None in records:
        raise ValueError('A group needs one build record hash.')
    rows = [{**row, 'turn': row.get('original_turn', row['turn'])} for report, _ in packets for row in report['rows']]
    result = metrics.summarize(rows)
    wall = sum(report['summary']['aggregate_wall_seconds'] for report, _ in packets)
    tokens = sum(len(row['token_ids']) for row in rows if row['status'] == 'ok')
    labels = [row.get('server_metrics', {}).get('session_label') for row in rows if row.get('server_metrics')]
    resources = [receipt.get('resources', {}) for _, receipt in packets]
    def maximum(key):
        values = [item[key] for item in resources if item.get(key) is not None]
        return max(values) if values else None
    available = [item['mem_available_floor_bytes'] for item in resources if item.get('mem_available_floor_bytes') is not None]
    return {**result, **coding.summarize(rows), 'aggregate_wall_seconds': wall,
            'aggregate_generation_tps': tokens / wall if wall else None,
            'aggregate_denominator': 'Sum of measured segment intervals. Queue waits and model loads are excluded.',
            'cache_labels': {key: labels.count(key) for key in sorted({value for value in labels if value is not None})},
            'segments': len(packets), 'lease_seconds': [receipt['lease_seconds'] for _, receipt in packets],
            'cache_budget_mib': sorted({receipt['cache_budget_mib'] for _, receipt in packets}),
            'fixed_state_bytes': maximum('source_derived_allocated_state_bytes'),
            'configured_capacity_tokens': maximum('configured_capacity_tokens'),
            'snapshot_cache_peak_bytes': maximum('snapshot_cache_peak_bytes'),
            'mem_available_floor_bytes': min(available) if available else None,
            'cache_evictions': None}


def collect(root):
    groups, segments = {}, {}
    for path in sorted(root.glob('*/report.json')):
        if '-probe-' in path.parent.name:
            continue
        receipt_path = path.parent / 'lease.json'
        if not receipt_path.exists():
            raise ValueError('A run report has no lease receipt.')
        match = re.fullmatch(r'.+-(serial|batch)-k([1248])-(greedy|sampled)(?:-s(\d+))?', path.parent.name)
        if match is None:
            raise ValueError('An unknown run label cannot prove its planned segment set.')
        report, receipt = json.loads(path.read_text()), json.loads(receipt_path.read_text())
        mode, count, policy, start = match.groups()
        key = '|'.join((mode, count, policy))
        groups[key] = [*groups.get(key, []), (report, receipt)]
        segments[key] = [*segments.get(key, []), None if start is None else int(start)]
    for key, actual in segments.items():
        count = int(key.split('|')[1])
        expected = [None] if count <= 2 else [0, 5, 10] if count == 4 else [0, 5, 7, 9, 11, 13]
        if len(actual) != len(expected) or set(actual) != set(expected):
            raise ValueError(f'The group {key} has an incomplete or duplicate segment set.')
    return {key: aggregate(packets) for key, packets in sorted(groups.items())}


def number(value, digits=2):
    return 'unknown' if value is None else f'{value:.{digits}f}'


def markdown(results):
    lines = ['# Agent session matrix', '', f'Completed mode/count/policy groups: {len(results)}/16. Missing groups remain pending.',
             'All times use seconds. All rates use tok/s.',
             'K4 and K8 rows use separate short segments. They are not uninterrupted fifteen-turn sessions.',
             'Later segment starts appear only in the separate cold-start category.', '',
             '| Mode | K | Policy | Turn type | n | TTFT p50 | TTFT p95 |',
             '|---|---:|---|---|---:|---:|---:|']
    for key, result in results.items():
        mode, count, policy = key.split('|')
        for category, value in result['turn_types'].items():
            times = value['ttft_seconds']
            if times['samples']:
                lines.append(f'| {mode} | {count} | {policy} | {category} | {times["samples"]} | {number(times["p50"])} | {number(times["p95"])} |')
    lines += ['', 'Aggregate generation includes queue and prefill within each segment. It excludes gaps between GPU leases.',
              'Per-session decode uses first-to-last token intervals. It includes interruptions from peer prefill.', '',
              '| Mode | K | Policy | Requests | Session decode min/max | Aggregate generation | Cache hit rate | Errors |',
              '|---|---:|---|---:|---:|---:|---:|---:|']
    for key, result in results.items():
        mode, count, policy = key.split('|')
        rates = [value for value in result['decode_tps_by_session'] if value is not None]
        rate = f'{number(min(rates))}/{number(max(rates))}' if rates else 'unknown'
        lines.append(f'| {mode} | {count} | {policy} | {result["completed_requests"]} | {rate} | {number(result["aggregate_generation_tps"])} | {number(result["cache_hit_rate"])} | {result["error_requests"] + result["rejected_requests"]} |')
    lines += ['', 'State bytes use the pinned source formula. Snapshot bytes and MemAvailable use measured observations.',
              'All memory columns use GiB, except the cache budget, which uses MiB.', '',
              '| Mode | K | Policy | Cache budget MiB | Fixed state GiB | Snapshot peak GiB | MemAvailable floor GiB | Leases s |',
              '|---|---:|---|---:|---:|---:|---:|---|']
    for key, result in results.items():
        mode, count, policy = key.split('|')
        def gib(field):
            return number(result[field] / (1 << 30)) if result[field] is not None else 'unknown'
        leases = ', '.join(number(value, 1) for value in result['lease_seconds'])
        budgets = ', '.join(str(value) for value in result['cache_budget_mib'])
        lines.append(f'| {mode} | {count} | {policy} | {budgets} | {gib("fixed_state_bytes")} | {gib("snapshot_cache_peak_bytes")} | {gib("mem_available_floor_bytes")} | {leases} |')
    lines += ['', 'Engine done events do not report eviction counts. Evictions stay unknown.',
              'Detailed cache labels remain in the JSON summary. Shared hits do not prove complete session restoration.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--markdown', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.root)
    runner.write_new(args.json, result)
    with args.markdown.open('x') as output:
        output.write(markdown(result))


if __name__ == '__main__': main()
