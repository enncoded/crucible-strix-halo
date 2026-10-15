"""Concurrent chat requests against an already running Crucible OpenAI endpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit
from . import metrics, stream, telemetry, workload


def tool_hashes():
    paths = [Path(module.__file__) for module in (metrics, stream, telemetry, workload)]
    paths = [Path(__file__), *paths, Path(workload.code.__code__.co_filename)]
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def validate_plan(plan):
    count = plan.get('concurrency')
    sessions = plan.get('sessions')
    if plan.get('schema') != 1 or type(count) is not int or count not in (1, 2, 4, 8):
        raise ValueError('The plan needs schema 1 and concurrency 1, 2, 4, or 8.')
    if not isinstance(sessions, list) or len(sessions) != count:
        raise ValueError('The plan needs one nonempty history per session.')
    for session, turns in enumerate(sessions):
        if not isinstance(turns, list) or not 1 <= len(turns) <= 1024:
            raise ValueError('Invalid session history length.')
        for index, turn in enumerate(turns):
            if (turn.get('session') != session or turn.get('turn') != index or
                    workload.digest(turn['body']) != turn.get('request_sha256') or
                    type(turn.get('prompt_tokens')) is not int or turn['prompt_tokens'] <= 0 or
                    turn['prompt_tokens'] + turn['body']['max_tokens'] > turn['context_limit_tokens']):
                raise ValueError('Invalid turn identity, hash, or context budget.')
            body = turn['body']
            if (body.get('stream') is not True or body.get('return_token_ids') is not True or
                    body.get('stream_options') != {'include_usage': True} or
                    type(body.get('max_tokens')) is not int or not 200 <= body['max_tokens'] <= 800):
                raise ValueError('Each turn needs streaming IDs, usage, and 200..800 output tokens.')


def checked_row(turn, row):
    extra = {key: value for key, value in turn.items() if key not in ('body', 'prompt_ids_sha256')}
    result = {**extra, **row}
    if result['status'] == 'ok':
        try:
            if result['usage']['prompt_tokens'] != turn['prompt_tokens']:
                raise ValueError('The server prompt count differs from the frozen fixture.')
            metrics.validate(result)
            return {**result, 'server_metrics': telemetry.validate(result.get('server_metrics'), result['usage'])}
        except (ValueError, KeyError, TypeError) as error:
            return {**extra, 'status': 'error', 'error_type': type(error).__name__,
                    'submitted_seconds': row['submitted_seconds'], 'finished_seconds': row['finished_seconds']}
    return result


def execute_session(turns, url, barrier, timeout, transport):
    barrier.wait(timeout=30)
    rows = []
    for turn in turns:
        row = transport(url, turn['body'], session=turn['session'], turn=turn['turn'], timeout=timeout)
        result = checked_row(turn, row)
        rows = [*rows, result]
        if result['status'] != 'ok':
            break
    return rows


def run(plan, url, *, label, timeout=300, decode_ceiling=None, transport=stream.request):
    parsed = urlsplit(url)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or
            parsed.query or parsed.fragment or parsed.path != '/v1/chat/completions'):
        raise ValueError('Use an HTTP chat endpoint without credentials, query, or fragment.')
    if not isinstance(label, str) or not 1 <= len(label) <= 120 or not 0 < timeout <= 900:
        raise ValueError('Use a baseline label and a timeout in (0,900] seconds.')
    before, plan_hash = tool_hashes(), workload.digest(plan)
    frozen = json.loads(json.dumps(plan, allow_nan=False))
    validate_plan(frozen)
    barrier = threading.Barrier(frozen['concurrency'])
    with ThreadPoolExecutor(max_workers=frozen['concurrency']) as pool:
        futures = [pool.submit(execute_session, turns, url, barrier, timeout, transport) for turns in frozen['sessions']]
        rows = [row for future in futures for row in future.result()]
    summary = metrics.summarize(rows, decode_ceiling=decode_ceiling)
    unchanged = before == tool_hashes() and workload.digest(plan) == plan_hash
    expected = sum(len(turns) for turns in frozen['sessions'])
    return {'schema': 1, 'status': 'PASS' if unchanged and summary['completed_requests'] == expected else 'FAIL',
            'label': label, 'endpoint': url, 'concurrency': frozen['concurrency'], 'plan_sha256': plan_hash,
            'tool_hashes': before, 'inputs_unchanged': unchanged, 'teacher_forced': True,
            'expected_requests': expected, 'skipped_requests': expected - len(rows), 'rows': rows, 'summary': summary}


def prepare(checkpoint, *, count, context, output, seed, server_metrics):
    from server.api import validate_request
    from server.tokenization import ModelTokenizer
    tokenizer = ModelTokenizer(checkpoint)
    def encode(body):
        local = {key: value for key, value in body.items() if key != 'crucible_metrics'}
        return tokenizer.encode(validate_request(local, 'chat', 'fixture'))
    def text(value):
        return tokenizer.tokenizer.encode(value, add_special_tokens=False).ids
    sessions = [workload.session(encode, text, session=n, context_tokens=context, output_tokens=output,
                                 seed=seed, metrics=server_metrics) for n in range(count)]
    paths = [Path(checkpoint) / 'tokenizer.json', Path(checkpoint) / 'tokenizer_config.json']
    template = Path(checkpoint) / 'chat_template.jinja'
    if template.exists(): paths = [*paths, template]
    return {'schema': 1, 'concurrency': count, 'context_limit_tokens': context, 'output_tokens': output,
            'seed': seed, 'teacher_forced': True, 'sessions': sessions,
            'tokenizer_hashes': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            'tool_hashes': tool_hashes()}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as output:
        output.write(json.dumps(value, allow_nan=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('prepare')
    build.add_argument('checkpoint', type=Path)
    build.add_argument('--sessions', type=int, choices=(1, 2, 4, 8), required=True)
    build.add_argument('--context', type=int, choices=(65536, 131072), required=True)
    build.add_argument('--tokens', type=int, default=400)
    build.add_argument('--seed', type=int, default=20261004)
    build.add_argument('--legacy-metrics', action='store_true')
    build.add_argument('--output', type=Path, required=True)
    measure = commands.add_parser('run')
    measure.add_argument('plan', type=Path)
    measure.add_argument('--url', required=True)
    measure.add_argument('--label', required=True)
    measure.add_argument('--timeout', type=float, default=300)
    measure.add_argument('--decode-ceiling', type=float)
    measure.add_argument('--server-record', type=Path, required=True)
    measure.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists(): raise ValueError('Use a new output path. Existing evidence is immutable.')
        if args.command == 'prepare':
            result = prepare(args.checkpoint, count=args.sessions, context=args.context, output=args.tokens,
                             seed=args.seed, server_metrics=not args.legacy_metrics)
        else:
            plan_bytes, record = args.plan.read_bytes(), args.server_record.read_bytes()
            plan = json.loads(plan_bytes, object_pairs_hook=stream.unique_pairs, parse_constant=stream.invalid_constant)
            result = run(plan, args.url, label=args.label, timeout=args.timeout, decode_ceiling=args.decode_ceiling)
            result = {**result, 'plan_file_sha256': hashlib.sha256(plan_bytes).hexdigest(),
                      'server_record_sha256': hashlib.sha256(record).hexdigest()}
            if args.plan.read_bytes() != plan_bytes or args.server_record.read_bytes() != record:
                result = {**result, 'status': 'FAIL', 'inputs_unchanged': False}
        write_new(args.output, result)
        if result.get('status', 'PASS') != 'PASS': parser.exit(1, 'The benchmark is incomplete. Inspect its saved report.\n')
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f'Agent benchmark failed: {type(error).__name__}. Check the inputs and output path.\n')


if __name__ == '__main__': main()
