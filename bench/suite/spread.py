#!/usr/bin/env python3
"""Summarize repeated fixed-suite reports without hiding individual measurements."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import run as suite

FIELDS = ('prefill_tps', 'ttft_seconds', 'decode_tps', 'acceptance_pct', 'decode_ceiling_pct', 'prefill_ceiling_pct')
META = ('tokenizer_sha256', 'config_sha256', 'quantization_sha256', 'prefill_rows')


def distribution(values):
    if all(value is None for value in values):
        return None
    if any(value is None or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('spread needs finite nonnegative metrics or consistent null values')
    return {'min': min(values), 'median': median(values), 'max': max(values)}


def unique_reports(reports):
    fingerprints = [hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest()
                    for report in reports]
    commands = []
    for report in reports:
        if 'queued_command' not in report:
            continue
        command = report['queued_command']
        if not isinstance(command,list) or not command or any(type(value) is not str for value in command):
            raise ValueError('invalid queued run identity')
        commands = [*commands, tuple(command)]
    if len(set(fingerprints)) != len(fingerprints) or len(set(commands)) != len(commands):
        raise ValueError('duplicate measurement report or queued run identity')


def groups_for(reports):
    if not reports:
        raise ValueError('no repetition reports')
    first, groups = reports[0], {}
    for report in reports:
        if (any(report.get(key) != first.get(key) for key in META) or
                report['build']['binary_sha256'] != first['build']['binary_sha256'] or
                report['build']['source_commit'] != first['build']['source_commit']):
            raise ValueError('repetition checkpoint or binary identity differs')
        if not report['rows']:
            raise ValueError('empty repetition report')
        suite.check_rows(report['rows'], report['rows'])
        for row in report['rows']:
            tokens = report['settings'].get('request_tokens', {}).get(row['name'], report['settings']['tokens'])
            suite.validate_result(row, {'tokens': row['prompt_tokens']}, tokens)
            key = row['name'], row['spec']
            sample = {'row':row, 'tokens':tokens, 'chunk':report['settings']['chunk'],
                      'mode':report['settings']['mode'], 'warmup':report['settings']['warmup']}
            groups[key] = [*groups.get(key, []), sample]
    return groups


def summarize(reports, samples):
    if samples < 2:
        raise ValueError('spread needs at least two measured repetitions')
    unique_reports(reports)
    groups = groups_for(reports)
    result = []
    for key, records in sorted(groups.items()):
        first = records[0]
        if len(records) != samples:
            raise ValueError(f'expected {samples} complete repetitions: {key}')
        for record in records:
            if (any(record[field] != first[field] for field in ('tokens','chunk','mode','warmup')) or
                    record['row']['prompt_sha256'] != first['row']['prompt_sha256'] or
                    record['row']['output_ids'] != first['row']['output_ids']):
                raise ValueError(f'repetition request or IDs differ: {key}')
        row = {'name':key[0], 'spec':key[1], 'samples':samples, 'prompt_sha256':first['row']['prompt_sha256'],
               'output_ids_sha256':hashlib.sha256(json.dumps(first['row']['output_ids']).encode()).hexdigest()}
        result.append({**row, **{field:distribution([record['row'][field] for record in records]) for field in FIELDS}})
    return {'schema':1, 'build':reports[0]['build'], 'checkpoint':reports[0]['checkpoint'],
            'samples':samples, 'rows':result}


def table(report):
    def numbers(value):
        return 'n/a' if value is None else ' / '.join(f'{value[key]:.3f}' for key in ('min','median','max'))
    lines = [f"# Repetition spread: {report['build']['source_commit']}", '',
             'Values use min / median / max. Each row includes every measured repetition.', '',
             '| Prompt | Spec | Samples | Prefill tok/s | TTFT s | Decode tok/s |',
             '|---|---:|---:|---|---|---|']
    for row in report['rows']:
        lines.append(f"| {row['name']} | {row['spec']} | {row['samples']} | {numbers(row['prefill_tps'])} | {numbers(row['ttft_seconds'])} | {numbers(row['decode_tps'])} |")
    return '\n'.join(lines) + '\n'


def read_reports(files, output):
    if output == output.with_suffix('.md'):
        raise ValueError('JSON and Markdown outputs need distinct paths')
    for target in (output, output.with_suffix('.md')):
        if target in files or (target.exists() and any(target.samefile(file) for file in files)):
            raise ValueError('spread output cannot alias a primary input report')
        if target.exists():
            raise ValueError('use new spread output files')
    payloads = [(file,file.read_bytes()) for file in files]
    hashes = {str(file):hashlib.sha256(payload).hexdigest() for file,payload in payloads}
    return [json.loads(payload) for _,payload in payloads], hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs='+')
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--expect-full-suite', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        files = [file.resolve() for file in args.reports]
        if len(files) != len(set(files)):
            raise ValueError('duplicate repetition report path')
        output = args.output.resolve()
        reports, hashes = read_reports(files, output)
        result = summarize(reports, args.samples)
        if args.expect_full_suite:
            expected = {(name,spec) for name in suite.DEFAULT_PROMPTS.split(',') for spec in (0,2)}
            expected |= {(f'long_{length}',0) for length in (4096,16384,65536)}
            if {(row['name'],row['spec']) for row in result['rows']} != expected:
                raise ValueError('full suite needs all four short prompts and all three long fixtures')
        if any(suite.digest(file) != hashes[str(file)] for file in files):
            raise ValueError('primary input report changed after it was read')
        result = {**result, 'inputs':hashes, 'spread_tool_sha256':suite.digest(Path(__file__))}
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as stream:
            stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
        with output.with_suffix('.md').open('x') as stream:
            stream.write(table(result))
        print(table(result))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'spread: {error}\n')


if __name__ == '__main__': main()
