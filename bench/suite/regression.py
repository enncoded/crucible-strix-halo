"""Short post-push checks against frozen published-main measurements."""
import argparse
import json
import math
import os
from pathlib import Path

import run as suite

DEFAULT_BASELINE = suite.HERE / 'results/2026-10-03-main-c286bda'
METADATA = ('tokenizer_sha256', 'config_sha256', 'quantization_sha256')
LONG = 'long_4096'


def load_baseline(root):
    manifest = json.loads((root/'qualification.json').read_text())
    reports = []
    for name in ('short', 'prefill-4096'):
        file = root/name/'results.json'
        if suite.digest(file) != manifest['reports'][name]:
            raise ValueError(f'published baseline report hash differs: {name}')
        reports.append(json.loads(file.read_text()))
    if any(reports[0][key] != reports[1][key] for key in METADATA):
        raise ValueError('published baseline checkpoint metadata differs between modes')
    if any(report['prefill_rows'] != 4096 or report['load'].get('draft_head') != 'full' for report in reports):
        raise ValueError('published baseline needs bulk4096 and full-head records')
    return {**reports[0], 'rows': [row for report in reports for row in report['rows']],
            'baseline_root': str(root.resolve()), 'baseline_report_hashes': manifest['reports']}


def indexed(report):
    rows = report['rows']
    keys = [(row['name'], row['spec']) for row in rows]
    if not rows or len(keys) != len(set(keys)):
        raise ValueError('missing or duplicate regression rows')
    for row in rows:
        for field in ('prefill_tps', 'ttft_seconds') + (('decode_tps',) if row['name'] != LONG else ()):
            if not math.isfinite(row[field]) or row[field] <= 0:
                raise ValueError(f'invalid regression metric: {field}')
        if not row['output_ids'] or any(type(v) is not int or not 0 <= v < 248320 for v in row['output_ids']):
            raise ValueError('invalid regression output IDs')
    return dict(zip(keys, rows))


def compare(current, baseline, noise):
    if not math.isfinite(noise) or not 0 <= noise <= 50:
        raise ValueError('noise percent must be finite in [0,50]')
    if any(current.get(key) != baseline.get(key) for key in METADATA):
        raise ValueError('regression checkpoint metadata differs from the published baseline')
    rows, reference = indexed(current), indexed(baseline)
    shorts = {name for name, _ in rows if name != LONG}
    expected = {(name, spec) for name in shorts for spec in (0, 2)} | {(LONG, 0)}
    if not 1 <= len(shorts) <= 2 or set(rows) != expected or not set(rows) <= set(reference):
        raise ValueError('regression needs one or two complete serial/spec2 pairs and 4K prefill')
    changes = []
    for key, row in rows.items():
        old = reference[key]
        if row['prompt_sha256'] != old['prompt_sha256'] or row['output_ids'] != old['output_ids']:
            raise ValueError(f'regression IDs differ from the published baseline: {key}')
        metric = 'prefill_tps' if row['name'] == LONG else 'decode_tps'
        rate = 100 * (row[metric] / old[metric] - 1)
        latency = 100 * (row['ttft_seconds'] / old['ttft_seconds'] - 1)
        changes.append({'name': key[0], 'spec': key[1], 'metric': metric, 'rate_change_pct': rate,
                        'ttft_change_pct': latency, 'status': 'FAIL' if rate < -noise or latency > noise else 'PASS'})
    return {'status': 'PASS' if all(row['status'] == 'PASS' for row in changes) else 'FAIL',
            'noise_percent': noise, 'rows': changes, 'baseline_root': baseline.get('baseline_root'),
            'baseline_report_hashes': baseline.get('baseline_report_hashes')}


def table(verdict):
    lines = ['# Post-push regression', '', f"Verdict: {verdict['status']}. Noise threshold: {verdict['noise_percent']:.2f}%.", '',
             '| Prompt | Spec | Rate metric | Rate change % | TTFT change % | Verdict |',
             '|---|---:|---|---:|---:|---|']
    for row in verdict['rows']:
        lines.append(f"| {row['name']} | {row['spec']} | {row['metric']} | {row['rate_change_pct']:+.2f} | {row['ttft_change_pct']:+.2f} | {row['status']} |")
    return '\n'.join(lines) + '\n'


def launch(args):
    if 'CRUCIBLE_PROFILE' in os.environ:
        raise ValueError('unset CRUCIBLE_PROFILE for regression measurements')
    baseline = load_baseline(args.baseline)
    corpus = suite.load_corpus(suite.HERE/'manifest.json')
    if suite.digest(args.checkpoint/'tokenizer.json') != corpus['tokenizer_sha256']:
        raise ValueError('checkpoint tokenizer differs from the fixed corpus')
    names = args.prompts.split(',')
    if len(set(names)) != len(names) or not 1 <= len(names) <= 2 or LONG in names:
        raise ValueError('select one or two unique short prompts')
    selected = [p for p in corpus['prompts'] if p['name'] in {*names, LONG}]
    if {p['name'] for p in selected} != {*names, LONG}:
        raise ValueError('unknown regression prompt name')
    compare({**baseline, 'rows':[row for row in baseline['rows'] if row['name'] in {*names, LONG}]},
            baseline, args.noise_percent)
    actual = {'tokenizer_sha256':corpus['tokenizer_sha256'],
              'config_sha256':suite.digest(args.checkpoint/'config.json'),
              'quantization_sha256':suite.digest(args.checkpoint/'quantization_config.json')}
    if any(actual[key] != baseline[key] for key in METADATA):
        raise ValueError('regression checkpoint metadata differs from the published baseline')
    # One load serves both decode modes and the prefill request.
    options = argparse.Namespace(**{**vars(args), 'tokens':160, 'chunk':8, 'spec':2, 'repeat':1, 'warmup':1,
                                   'mode':'regression', 'request_tokens':{LONG:1}, 'prefill_names':(LONG,),
                                   'request_warmup':{LONG:0}, 'serial_ceiling':None,
                                   'prefill_ceiling':2310, 'spec_ceiling':None})
    short_prompts = [p for p in selected if p['name'] != LONG]
    estimate = suite.check_budget(short_prompts, 160, 1, 1, 1000) + 4096/1000 + 1/20
    if estimate > 150:
        raise ValueError('post-push regression exceeds its 150-second admission budget')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    suite.save(out/'regression.json', {'status':'FAIL', 'error':'measurement or comparison incomplete'})
    report = suite.run_binary(args.binary, out/'a', selected, corpus, options, 4096)
    verdict = {**compare(report, baseline, args.noise_percent), 'estimated_seconds':estimate,
               'current_report_sha256':suite.digest(out/'a/results.json'),
               'regression_tool_sha256':suite.digest(Path(__file__))}
    suite.save(out/'regression.json', verdict)
    (out/'regression.md').write_text(table(verdict))
    print(table(verdict))
    if verdict['status'] != 'PASS':
        raise ValueError(f'performance regression exceeds {args.noise_percent}% noise threshold')


def add_parser(commands):
    parser = commands.add_parser('regression', help=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompts', default='code_fibonacci,copy_python')
    parser.add_argument('--noise-percent', type=float, default=5)
    parser.add_argument('--gpu-run', dest='gpu_run', type=Path, default=suite.work_paths.gpu_run())
    parser.add_argument('--gpu-wait', type=suite.positive, default=7200)
