"""Run only inside gpu-run. One temporary server and a bounded agent plan."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen
from . import coding, runner, stream

URL = 'http://127.0.0.1:8790'


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def verify_binary(path, record):
    value = sha(path)
    if value != record.get('binary_sha256'):
        raise ValueError('The binary does not match its build record.')
    return value


def server_hashes():
    return {str(path): sha(path) for path in sorted(Path('server').glob('*.py'))}


def require_idle_port(port=8790):
    with socket.socket() as probe:
        try:
            probe.bind(('127.0.0.1', port))
        except OSError as error:
            raise ValueError('Port 8790 is occupied. Do not measure another server.') from error


def engine_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def verify_done(path, rows, *, alive):
    if not alive:
        raise ValueError('The temporary server stopped before cleanup.')
    good = [row for row in rows if row['status'] == 'ok']
    done = [event for event in engine_events(path) if event.get('event') == 'done']
    expected = Counter((row['usage']['prompt_tokens'], row['usage']['completion_tokens']) for row in good)
    observed = Counter((event['prompt_tokens'], event['completion_tokens']) for event in done)
    ids = {event['request_id'] for event in done if 'request_id' in event}
    if expected != observed or any((row.get('server_metrics') or {}).get('request_id') not in ids for row in good if ids):
        raise ValueError('The engine done events do not prove the measured responses.')


def mem_available():
    return next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def cache_budget(available):
    return min(16384, max(0, (available - (9 << 30)) // (1 << 20)))


def wait_ready(server, deadline, events, log):
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError('The temporary server stopped before readiness.')
        try:
            with urlopen(URL + '/v1/models', timeout=1) as response:
                own_ready = any(event.get('event') == 'ready' for event in engine_events(events))
                bound = log.exists() and 'Crucible is ready at http://127.0.0.1:8790/v1' in log.read_text()
                if response.status == 200 and own_ready and bound and server.poll() is None:
                    return time.monotonic()
        except (URLError, OSError):
            time.sleep(.2)
    raise TimeoutError('The temporary server did not become ready.')


def trace_poll(stop, path):
    since = 0
    with path.open('x') as output:
        while not stop.is_set():
            try:
                with urlopen(URL + f'/v1/crucible/trace?since={since}', timeout=2) as response:
                    value = json.load(response)
                events = [row for row in value['data'] if row['t_end'] > since]
                if events:
                    output.write(json.dumps(events, allow_nan=False) + '\n')
                    output.flush()
                    since = max(row['t_end'] for row in events)
            except (URLError, OSError, ValueError, KeyError) as error:
                output.write(json.dumps({'poll_error': type(error).__name__}) + '\n')
                output.flush()
            available = mem_available()
            output.write(json.dumps({'mem_available_bytes': available, 'observed_monotonic': time.monotonic()}) + '\n')
            output.flush()
            stop.wait(1)


def server_command(args, plan, wrapper):
    command = [sys.executable, '-m', 'server', str(args.checkpoint), '--engine', str(wrapper),
               '--ctx', '65536', '--conversation-cache-mib', str(args.cache_mib), '--prefill', '4096']
    # A missing --streams is --streams 1 (batched) since an earlier change, so the serial arm must say --serial.
    return command + (['--streams', str(plan['concurrency'])] if args.mode == 'batch' else ['--serial'])


def measure(args):
    started = time.monotonic()
    plan_bytes, record_bytes = args.plan.read_bytes(), args.record.read_bytes()
    plan, record = json.loads(plan_bytes), json.loads(record_bytes)
    runner.validate_plan(plan)
    binary_hash = verify_binary(args.engine, record)
    sources = server_hashes()
    args.output.mkdir(parents=True, exist_ok=False)
    wrapper = (args.output / 'engine-wrapper').resolve()
    wrapper.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' -m bench.agents.engine_tee "$@"\n')
    wrapper.chmod(0o700)
    environment = {**os.environ, 'AGENTS_ENGINE': str(args.engine.resolve()),
                   'AGENTS_EVENTS': str((args.output / 'engine-events.jsonl').resolve())}
    command = server_command(args, plan, wrapper)
    receipt = {'started_monotonic': started, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'limit_seconds': args.limit, 'priority': 2, 'command': command, 'mode': args.mode,
               'cache_budget_mib': args.cache_mib, 'capacity_probe': args.probe,
               'binary_sha256': binary_hash, 'server_hashes': sources,
               'engine_tee_sha256': sha(Path(__file__).with_name('engine_tee.py')),
               'plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
               'build_record_sha256': hashlib.sha256(record_bytes).hexdigest(), 'environment': {
                   key: value for key, value in os.environ.items() if key.startswith('CRUCIBLE_')}}
    stop, server, poller = threading.Event(), None, None
    with (args.output / 'server.log').open('x') as log:
        try:
            require_idle_port()
            server = subprocess.Popen(command, stdout=log, stderr=log, env=environment)
            ready = wait_ready(server, min(started + 100, started + args.limit - 20),
                               args.output / 'engine-events.jsonl', args.output / 'server.log')
            receipt['ready_seconds'] = ready - started
            receipt['ready_mem_available_bytes'] = mem_available()
            receipt['recommended_cache_mib'] = cache_budget(receipt['ready_mem_available_bytes'])
            poller = threading.Thread(target=trace_poll, args=(stop, args.output / 'trace.jsonl'))
            poller.start()
            def transport(url, body, **kwargs):
                remaining = started + args.limit - 20 - time.monotonic()
                if remaining <= 0:
                    row = {'status': 'error', 'error_type': 'LeaseDeadline',
                           'submitted_seconds': time.monotonic(), 'finished_seconds': time.monotonic()}
                else:
                    row = stream.request(url, body, **{**kwargs, 'timeout': min(kwargs['timeout'], remaining)})
                if row['status'] != 'ok' and server.poll() is None:
                    server.terminate()
                return row
            result = {'schema': 1, 'status': 'PASS', 'rows': []} if args.probe else runner.run(
                plan, URL + '/v1/chat/completions', label=args.label, timeout=args.limit - 20, transport=transport)
            receipt['server_alive_after_measurement'] = server.poll() is None
            verify_done(args.output / 'engine-events.jsonl', result['rows'], alive=receipt['server_alive_after_measurement'])
            receipt['measurement_seconds'] = time.monotonic() - ready
            result = {**result, 'coding_summary': coding.summarize(result['rows']),
                      'binary_sha256': binary_hash, 'build_record': record, 'server_hashes': sources}
            runner.write_new(args.output / 'report.json', result)
            receipt['status'] = result['status']
        except Exception as error:
            receipt.update(status='FAIL', error_type=type(error).__name__)
        finally:
            stop.set()
            if poller is not None:
                poller.join(timeout=3)
            if server is not None:
                if server.poll() is None:
                    server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=2)
                receipt['server_returncode'] = server.returncode
            receipt['resources'] = coding.resources(args.output)
            receipt['lease_seconds'] = time.monotonic() - started
            receipt['inputs_unchanged'] = (args.plan.read_bytes() == plan_bytes and args.record.read_bytes() == record_bytes
                                           and sha(args.engine) == binary_hash and server_hashes() == sources)
            if not receipt['inputs_unchanged']:
                receipt['status'] = 'FAIL'
            runner.write_new(args.output / 'lease.json', receipt)
    return 0 if receipt['status'] == 'PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('serial', 'batch'), required=True)
    parser.add_argument('--limit', type=int, choices=(300, 600), default=300)
    parser.add_argument('--cache-mib', type=int, choices=range(16385), default=0)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    return measure(args)


if __name__ == '__main__': raise SystemExit(main())
