#!/usr/bin/env python3
"""Serial serve against `--streams 1` serve: the same request set, the same token IDs (train 19).

The retirement of the serial stack needs proof that a batch server with one stream
answers every request like the serial server. This tool sends one fixed set (tools/serve_parity_cases.py: greedy,
sampled with seeds, speculative, vision, session replay, and the known gaps) to both servers and compares the token
IDs, the finish reason and the token counts. The cache fields (cached_tokens) and the speed are only reported.

  serve_parity.py plan                         print the cases and the GPU estimate
  serve_parity.py run --serial-url U --batch-url U --out report.json    two servers that already run
  serve_parity.py launch ENGINE CHECKPOINT --out report.json            start each server in turn, then stop it

`launch` runs the serial server, then the `--streams 1` server, one at a time, in one process: wrap it in one
`gpu-run` lease. It stops only the processes that it started. Exit code 0 means no FAIL.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.serve_parity_cases import GROUPS, default_cases

REQUEST_SECONDS = 240
READY_SECONDS = 240
# Planning rates for `plan`.
PREFILL_RATE, DECODE_RATE, LOAD_SECONDS = 1000.0, 35.0, 60.0
# Kill line of multi-session gate G6: a lone batch decoder below 95 percent of serial speed.
SLOW_RATIO = 0.95


def post(base, body, timeout=REQUEST_SECONDS):
    """One non-stream chat request. Returns a result dict; an HTTP error is a result, not an exception."""
    request = urllib.request.Request(base.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
        status = 200
    except urllib.error.HTTPError as error:
        try:
            data = json.load(error)
        except ValueError:
            data = {}
        return dict(status=error.code, error=str(data.get("error", data))[:200], seconds=time.monotonic() - started)
    except (urllib.error.URLError, OSError, ValueError) as error:
        # Status 0: no answer. The case fails unless the other server fails the same way.
        return dict(status=0, error=type(error).__name__, seconds=time.monotonic() - started)
    choice = data["choices"][0]
    metrics = (data.get("usage") or {}).get("crucible_metrics") or {}
    return dict(status=status, ids=choice.get("token_ids"), finish=choice.get("finish_reason"),
                text=(choice.get("message") or {}).get("content"), usage=data.get("usage") or {},
                cached=metrics.get("restored_tokens"), label=metrics.get("session_label"),
                seconds=time.monotonic() - started)


def run_cases(base, cases, *, vision, cache, send=post):
    """Send every selected case in order to one server. Cases a server cannot take are skipped, not failed."""
    results = {}
    for case in cases:
        if (case.needs_vision and not vision) or (case.needs_cache and not cache):
            continue
        results[case.name] = send(base, case.body_for(results))
    return results


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def judge(case, serial, batch):
    """PASS, FAIL, KNOWN_GAP or GAP_CLOSED for one case, with the evidence."""
    both_ok = serial["status"] == 200 and batch["status"] == 200
    if case.gap and serial["status"] == 200 and batch["status"] == 400:
        return dict(verdict="KNOWN_GAP", note=batch.get("error"))
    if not both_ok:
        same = serial["status"] == batch["status"]
        return dict(verdict="PASS" if same else "FAIL", note=f"HTTP {serial['status']} serial, {batch['status']} batch")
    ids_a, ids_b = serial.get("ids"), batch.get("ids")
    if not isinstance(ids_a, list) or not isinstance(ids_b, list):
        return dict(verdict="FAIL", note="a response has no token_ids")
    where = first_difference(ids_a, ids_b)
    usage = [(r["usage"].get("prompt_tokens"), r["usage"].get("completion_tokens")) for r in (serial, batch)]
    same = where is None and serial["finish"] == batch["finish"] and usage[0] == usage[1]
    if case.gap:
        return dict(verdict="GAP_CLOSED" if same else "FAIL", first_difference=where)
    return dict(verdict="PASS" if same else "FAIL", first_difference=where,
                note=None if same else f"finish {serial['finish']}/{batch['finish']} usage {usage[0]}/{usage[1]}")


def speed(serial, batch):
    """Completion tokens per wall second, batch over serial. Information only: one run is not an ABAB timing."""
    def rate(result):
        tokens = (result.get("usage") or {}).get("completion_tokens") or 0
        return tokens / result["seconds"] if result.get("seconds") and tokens > 4 else None
    a, b = rate(serial), rate(batch)
    return round(b / a, 3) if a and b else None


def compare(cases, serial_results, batch_results):
    rows = []
    for case in cases:
        if case.name not in serial_results or case.name not in batch_results:
            continue
        serial, batch = serial_results[case.name], batch_results[case.name]
        row = dict(name=case.name, group=case.group, **judge(case, serial, batch))
        row.update(cached=[serial.get("cached"), batch.get("cached")],
                   cache_label=[serial.get("label"), batch.get("label")], speed_ratio=speed(serial, batch))
        rows.append(row)
    verdicts = [row["verdict"] for row in rows]
    summary = {kind: verdicts.count(kind) for kind in ("PASS", "FAIL", "KNOWN_GAP", "GAP_CLOSED")}
    slow = [row["name"] for row in rows if row["speed_ratio"] is not None and row["speed_ratio"] < SLOW_RATIO]
    return dict(status="FAIL" if summary["FAIL"] or not rows else "PASS", summary=summary, slow_cases=slow, cases=rows)


def estimate(cases, *, vision=True, cache=True):
    """Expected GPU seconds of one `launch` run: two loads, then both servers run every case once."""
    seconds = 0.0
    for case in cases:
        if (case.needs_vision and not vision) or (case.needs_cache and not cache):
            continue
        seconds += case.prompt_tokens / PREFILL_RATE + case.gen_tokens / DECODE_RATE + 0.5
    return round(2 * (LOAD_SECONDS + seconds))


def wait_ready(base, process, timeout=READY_SECONDS):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"the server stopped with exit code {process.returncode}")
        try:
            with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=5):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    raise TimeoutError("the server did not become ready")


def serve_args(args, role, port):
    command = shlex.split(args.server_command) + [str(args.checkpoint), "--engine", str(args.engine),
                                                  "--port", str(port), "--ctx", str(args.ctx), "--prefill", "4096"]
    if args.cache_mib:
        command += ["--conversation-cache-mib", str(args.cache_mib), "--session-replay", "1"]
    if args.vision:
        command += ["--vision"]
    # A missing --streams means --streams 1 since train 19, so the serial role needs --serial.
    command += ["--streams", "1"] if role == "batch" else ["--serial"]
    return command + shlex.split(args.extra)


def launch_one(args, role, port, cases):
    """Start one server, run the set, stop that process. The log goes beside the report."""
    log = open(args.out.with_suffix(f".{role}.log"), "w")
    process = subprocess.Popen(serve_args(args, role, port), stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}/v1"
    try:
        wait_ready(base, process)
        return run_cases(base, cases, vision=args.vision, cache=bool(args.cache_mib))
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
        log.close()


def select(groups):
    """The chosen groups, in set order. Replay turns need their first turn, so the replay group stays whole."""
    return [case for case in default_cases() if not groups or case.group in groups]


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=report["status"], **report["summary"], slow_cases=report["slow_cases"])))
    return 0 if report["status"] == "PASS" else 1


def parser_for():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "launch"):
        sub = commands.add_parser(name)
        sub.add_argument("--group", action="append", choices=GROUPS, help="Select case groups. May repeat.")
        sub.add_argument("--vision", action="store_true", help="Servers run with --vision; adds the vision cases.")
        sub.add_argument("--cache-mib", type=int, default=2048, help="Conversation cache; 0 skips the replay cases.")
        if name != "plan":
            sub.add_argument("--out", type=Path, required=True)
    commands.choices["run"].add_argument("--serial-url", required=True)
    commands.choices["run"].add_argument("--batch-url", required=True)
    launch = commands.choices["launch"]
    launch.add_argument("engine", type=Path)
    launch.add_argument("checkpoint", type=Path)
    launch.add_argument("--ctx", type=int, default=32768)
    launch.add_argument("--port-base", type=int, default=8830, help="Serial uses this port, batch the next one.")
    launch.add_argument("--server-command", default=f"{sys.executable} -m server")
    launch.add_argument("--extra", default="", help="Extra server flags for both servers, as one string.")
    return parser


def main(argv=None):
    args = parser_for().parse_args(argv)
    cases = select(args.group)
    if args.command == "plan":
        for case in cases:
            print(f"{case.group:8} {case.name}")
        print(f"estimated GPU seconds: {estimate(cases, vision=args.vision, cache=bool(args.cache_mib))}")
        return 0
    cache = bool(args.cache_mib)
    if args.command == "run":
        serial = run_cases(args.serial_url, cases, vision=args.vision, cache=cache)
        batch = run_cases(args.batch_url, cases, vision=args.vision, cache=cache)
    else:
        serial = launch_one(args, "serial", args.port_base, cases)
        batch = launch_one(args, "batch", args.port_base + 1, cases)
    return write_report(args.out, compare(cases, serial, batch))


if __name__ == "__main__":
    raise SystemExit(main())
