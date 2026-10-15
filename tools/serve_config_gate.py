#!/usr/bin/env python3
"""Capture bounded native serve shards, then compare the complete configuration matrix."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.tokenization import ModelTokenizer
from tools.serve_invariance.capture import capture_cases
from tools.serve_invariance.model import CONFIGS, PACKED_OPS, compare, digest, file_digest, load_cases

DEFAULT_CORPUS = ROOT / "bench/suite/serve_invariance/manifest.json"


def save(path, report):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".config-gate-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def reserve(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write('{"status":"PENDING"}\n')


def parser_for():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="Run one bounded configuration shard under gpu-run.")
    capture.add_argument("engine", type=Path)
    capture.add_argument("checkpoint", type=Path)
    capture.add_argument("output", type=Path)
    capture.add_argument("--config", choices=CONFIGS, required=True)
    capture.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    capture.add_argument("--case", action="append", help="Select fixed request names. May repeat.")
    capture.add_argument("--group", choices=("bundle-a", "bundle-b", "synthetic"))
    capture.add_argument("--ctx", type=int, default=65536)
    capture.add_argument("--prefill", type=int, default=4096)
    capture.add_argument("--cache-mib", type=int, default=16384)
    capture.add_argument("--streams", type=int, default=4, choices=range(4, 9))
    capture.add_argument("--request-timeout", type=float, default=240)
    capture.add_argument("--packed-op", choices=PACKED_OPS, default="verify_step",
                         help="packed batch op; use step for an engine before G6d")
    capture.add_argument("--prefill-interleave", type=int, default=10,
                         help="batch configs: decode passes every N layers of a prompt chunk (the served default 10); "
                              "0 keeps whole chunks (required with --packed-op step)")
    capture.add_argument("--session-replay", type=int, choices=(0, 1), default=0,
                         help="1: cache-on configs run with session replay (the serve default); fresh requests "
                              "must still match cache-off")
    comparison = commands.add_parser("compare", help="Fail unless all configurations and fixed requests match.")
    comparison.add_argument("captures", type=Path, nargs="+")
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    return parser


def required_corpus(path):
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != 1 or len(manifest["cases"]) != 12:
        raise ValueError("The comparison needs all twelve fixed requests.")
    for name, expected in manifest["files"].items():
        source = path.parent / name
        if source.resolve().parent != path.parent.resolve() or file_digest(source) != expected:
            raise ValueError(f"A frozen corpus file changed: {name}")
    required = {case["name"]: case["request_sha256"] for case in manifest["cases"]}
    if len(required) != 12:
        raise ValueError("The fixed corpus contains duplicate requests.")
    prompts = {case["name"]: digest(json.loads((path.parent / case["prompt_ids_file"]).read_text()))
               for case in manifest["cases"] if case.get("prompt_ids_file")}
    return required, prompts


def main(argv=None):
    parser = parser_for()
    args = parser.parse_args(argv)
    reserved = False
    try:
        reserve(args.output)
        reserved = True
        if args.command == "compare":
            required, prompts = required_corpus(args.corpus)
            report = compare([json.loads(path.read_text()) for path in args.captures], required=required,
                             required_prompts=prompts)
        else:
            if args.case and args.group:
                raise ValueError("Select request names or one group, not both.")
            cases = load_cases(args.corpus, ModelTokenizer(args.checkpoint))
            selected = args.case or ([case.name for case in cases if case.group == args.group] if args.group else None)
            report = capture_cases(args.engine.resolve(), args.checkpoint.resolve(), cases, args.config,
                                   context=args.ctx, prefill=args.prefill, cache_mib=args.cache_mib,
                                   streams=args.streams, timeout=args.request_timeout, selected=selected,
                                   save=lambda value: save(args.output, value), packed_op=args.packed_op,
                                   session_replay=bool(args.session_replay),
                                   prefill_interleave=args.prefill_interleave)
        save(args.output, report)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        # A partial capture can never become PASS during comparison.
        if reserved and args.output.read_text() == '{"status":"PENDING"}\n':
            save(args.output, dict(status="FAIL", error=str(error)))
        print(f"config-invariance gate: {error}", file=sys.stderr)
        return 1
    print(json.dumps(dict(status=report["status"], error=report.get("error"))))
    for request in report.get("requests", []):
        print(f"{request['name']}: {'PASS' if request['match'] else 'FAIL'}")
        for row in request["observations"]:
            if not row["match"]:
                print(f"  {row['config']} {row['phase']}: token {row['first_difference']}, "
                      f"reference={row['reference_id']}, actual={row['actual_id']}")
    return 0 if report["status"] in ("PASS", "CAPTURED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
