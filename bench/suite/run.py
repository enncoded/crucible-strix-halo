#!/usr/bin/env python3
"""Build, run, and compare resident benchmark binaries. Python standard library only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone
from rooflines import annotate
from runtime_environment import snapshot as environment_snapshot, require_known, policy_hashes
sys.path.append(str(Path(__file__).resolve().parents[2] / 'tools'))  # tools/work_paths.py
import work_paths  # noqa: E402

RUNNER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
HERE = Path(__file__).resolve().parent
DEFAULT_PROMPTS = "code_fibonacci,chat_planning,prose_story,copy_python"
SERIAL_305 = 245.5e9 / 4615344324


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_corpus(path):
    corpus = json.loads(path.read_text())
    if corpus["schema"] != 1:
        raise ValueError("unsupported corpus schema")
    names = set()
    for prompt in corpus["prompts"]:
        file = path.parent / prompt["ids_file"]
        if digest(file) != prompt["sha256"]:
            raise ValueError(f"ID hash differs: {prompt['name']}")
        ids = [int(x) for x in file.read_text().strip().split(",")]
        if len(ids) != prompt["tokens"] or not ids or any(x < 0 or x >= 248320 for x in ids):
            raise ValueError(f"invalid IDs: {prompt['name']}")
        if prompt["name"] in names:
            raise ValueError("duplicate prompt name")
        names.add(prompt["name"])
    return corpus


def metrics(row, serial_ceiling, prefill_ceiling, spec_ceiling):
    output_tokens = len(row["output_ids"])
    pp = row["prompt_tokens"] / row["prefill_seconds"]
    decode = (output_tokens - 1) / row["decode_seconds"] if output_tokens > 1 else None
    ceiling = spec_ceiling if row["spec"] else serial_ceiling
    return {**row, "prefill_tps": pp, "decode_tps": decode,
            "acceptance_pct": 100 * row["accepted"] / row["proposed"] if row["proposed"] else None,
            "tokens_per_forward": (output_tokens - 1) / row["forwards"] if row["forwards"] else None,
            "decode_ceiling_pct": 100 * decode / ceiling if ceiling and decode else None,
            "prefill_ceiling_pct": 100 * pp / prefill_ceiling if prefill_ceiling else None}


def key(row):
    return row["name"], row["spec"], row["repeat"]


def check_rows(rows, expected):
    keys = [key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate result row")
    if set(keys) != {key(row) for row in expected}:
        raise ValueError("missing or unexpected result rows")
    serial = {(r["name"], r["repeat"]): r["output_ids"] for r in rows if r["spec"] == 0}
    for row in rows:
        if row["output_ids"] != serial[(row["name"], row["repeat"])]:
            raise ValueError(f"serial and speculative IDs differ: {row['name']}")


def check_budget(prompts, tokens, repeats, warmups, prefill_floor, modes=2):
    # This is admission control, not a performance claim. timeout remains mandatory.
    seconds = 35 + modes * (repeats + warmups) * sum(
        p["tokens"] / prefill_floor + tokens / 20 for p in prompts)
    if seconds > 270:
        raise ValueError(f"estimated run {seconds:.0f}s exceeds the 270s budget; select fewer prompts, "
                         "or set --prefill-floor to a measured lower bound for this binary")
    return seconds


def compare(a, b):
    if any(a.get(field) != b.get(field) for field in
           ("config_sha256", "quantization_sha256", "tokenizer_sha256", "settings")):
        raise ValueError("comparison request settings or checkpoint metadata differ")
    left = {key(row): row for row in a["rows"]}
    right = {key(row): row for row in b["rows"]}
    if left.keys() != right.keys():
        raise ValueError("comparison request sets differ")
    result = []
    for identity, old in left.items():
        new = right[identity]
        if old.get("prompt_sha256") != new.get("prompt_sha256"):
            raise ValueError("comparison prompt IDs differ")
        if old["output_ids"] != new["output_ids"]:
            raise ValueError(f"comparison output IDs differ: {identity}")
        result.append({"name": identity[0], "spec": identity[1], "repeat": identity[2],
                       "decode_change_pct": 100 * (new["decode_tps"] / old["decode_tps"] - 1)
                       if old["decode_tps"] else None,
                       "prefill_change_pct": 100 * (new["prefill_tps"] / old["prefill_tps"] - 1),
                       "ttft_change_pct": 100 * (new["ttft_seconds"] / old["ttft_seconds"] - 1)})
    return result


def comparison_table(rows):
    lines = ["| Prompt | Spec | Repeat | Decode change % | Prefill change % | TTFT change % |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [row["name"], str(row["spec"]), str(row["repeat"])] + [
            "n/a" if row[field] is None else f"{row[field]:+.2f}" for field in
            ("decode_change_pct", "prefill_change_pct", "ttft_change_pct")]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def save_comparison(path, a, b):
    rows = compare(a, b)
    save(path, rows)
    path.with_suffix(".md").write_text(comparison_table(rows))


def table(report):
    def number(value):
        return "n/a" if value is None else f"{value:.2f}"
    lines = [f"# {report['build']['source_branch']} @ {report['build']['source_commit']}", "",
             "| Prompt | Spec | Repeat | Prefill tok/s | TTFT s | Decode tok/s | Accept % | Prefill ceiling % | Decode ceiling % | Fused prefill ceiling % |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["rows"]:
        values = [row["name"], str(row["spec"]), str(row["repeat"])] + [number(row[x]) for x in
                  ("prefill_tps", "ttft_seconds", "decode_tps", "acceptance_pct",
                   "prefill_ceiling_pct", "decode_ceiling_pct")]
        values.append(number(row.get("prefill_fused_ceiling_pct")))
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "TTFT is engine prefill time through the first target ID. It excludes model load and request setup.",
              "Decode excludes the first output ID. No stop token ends the fixed output length.",
              "Unknown ceilings remain n/a. Percentages are analytical throughput ratios, not hardware counter measurements."]
    return "\n".join(lines) + "\n"


def build(args):
    source = args.source.resolve()
    binary = args.output.resolve()
    def git(*values):
        return subprocess.check_output(["git", "-C", str(source), *values], text=True).strip()
    if "RunGen(" not in (source / "engine/cli/gen_cli.h").read_text():
        raise ValueError("source needs the engine's RunGen interface")
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [args.hipcc, "-O3", "-std=c++20", "--offload-arch=gfx1151", "--rocm-path=/opt/rocm",
               "-I/opt/rocm/include", f'-DCRUCIBLE_ENGINE_HEADER="{source}/engine/engine.hip"',
               *(["-DCRUCIBLE_SUITE_PREFILL=1"] if getattr(args, "bulk_prefill", False) else []),
               str(HERE / "resident.hip"), "-lpthread", "-o", str(binary)]
    if bool(args.source_commit) != bool(args.source_branch):
        raise ValueError("set both --source-commit and --source-branch for an exported source tree")
    branch = args.source_branch or git("branch", "--show-current")
    commit = args.source_commit or git("rev-parse", "HEAD")
    sources = sorted([*source.glob("engine/**/*"), *source.glob("kernels/**/*"), source / "engine/exl3/safetensors.h"])
    metadata = {"source_branch": branch, "source_commit": commit,
                "bulk_prefill": getattr(args, "bulk_prefill", False),
                "source_dirty": None if args.source_commit else bool(git("status", "--porcelain", "--", "engine", "kernels", "engine/exl3/safetensors.h")),
                "source_identity": "operator-supplied archive" if args.source_commit else "git worktree",
                "source_hashes": {str(p.relative_to(source)): digest(p) for p in sources if p.is_file()},
                "harness_sha256": digest(HERE / "resident.hip"),
                "harness_headers": {"full_head_policy.h": digest(HERE / "full_head_policy.h")},
                "generation_policy": {"draft_head": "full", "draft_q": 0, "lookup": 0}, "command": command,
                "compiler": subprocess.check_output([args.hipcc, "--version"], text=True)}
    subprocess.run(command, check=True)
    save(Path(str(binary) + ".build.json"), {**metadata, "binary_sha256": digest(binary)})


def worker():
    binary, checkpoint, requests, ctx, snapshot, *prefill = sys.argv[2:]
    os.environ.setdefault("HSA_USERPTR_FOR_PAGED_MEM", "0")
    state = {"utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
             "platform": platform.platform(), "meminfo": Path("/proc/meminfo").read_text(),
             "buddyinfo": Path("/proc/buddyinfo").read_text(),
             **environment_snapshot(os.environ)}
    save(Path(snapshot), state)
    require_known(state)
    os.execv(binary, [binary, checkpoint, requests, ctx, *prefill])


def requests_file(path, prompts, args):
    expected, lines = [], []
    for prompt in prompts:
        ids = str((HERE / prompt["ids_file"]).resolve())
        tokens = getattr(args, 'request_tokens', {}).get(prompt['name'], args.tokens)
        warmup = getattr(args, 'request_warmup', {}).get(prompt['name'], args.warmup)
        prefill = getattr(args, 'mode', 'decode') == 'prefill' or prompt['name'] in getattr(args, 'prefill_names', ())
        for repeat in range(-warmup, args.repeat):
            for spec in ((0,) if prefill else (0, args.spec)):
                lines.append(f'{prompt["name"]} {json.dumps(ids)} {tokens} {args.chunk} {spec} {max(-1, repeat)}')
                if repeat >= 0:
                    expected.append({"name": prompt["name"], "spec": spec, "repeat": repeat})
    path.write_text("\n".join(lines) + "\n")
    return expected


def validate_result(row, prompt, tokens):
    if len(row["output_ids"]) != tokens or row["prompt_tokens"] != prompt["tokens"]:
        raise ValueError("result token count differs from the request")
    if any(not math.isfinite(row[k]) or row[k] <= 0 for k in ("prefill_seconds", "ttft_seconds")) or \
            not math.isfinite(row["decode_seconds"]) or row["decode_seconds"] < 0 or \
            (tokens > 1 and row["decode_seconds"] == 0):
        raise ValueError("invalid timing")
    if not 0 <= row["accepted"] <= row["proposed"] or row["forwards"] < int(tokens > 1):
        raise ValueError("invalid draft counters")
    if any(type(value) is not int or not 0 <= value < 248320 for value in row["output_ids"]):
        raise ValueError("invalid output IDs")


def run_binary(binary, out, prompts, corpus, args, prefill_rows=0):
    priority = getattr(args, 'priority', 3)
    if type(priority) is not int or not 0 <= priority <= 3:
        raise ValueError('GPU priority must be an integer from zero through three')
    binary = binary.resolve()
    build_info = json.loads(Path(str(binary) + ".build.json").read_text())
    if digest(binary) != build_info["binary_sha256"]:
        raise ValueError("binary differs from its build metadata; rebuild it with this runner")
    if prefill_rows and not build_info.get("bulk_prefill", False):
        raise ValueError("binary needs --bulk-prefill at build time for this request")
    out.mkdir(parents=True, exist_ok=False)
    environment = environment_snapshot(os.environ)
    save(out/'environment.json', environment)
    require_known(environment)
    expected = requests_file(out / "requests.txt", prompts, args)
    context = max(4096, max(p["tokens"] for p in prompts) + args.tokens + args.spec + 8)
    command = [str(args.gpu_run), "-p", str(priority), "-w", str(getattr(args, "gpu_wait", 3600)), "--", "timeout", "-k", "10", "300",
               sys.executable, str(HERE / "run.py"), "_worker", str(binary), str(args.checkpoint.resolve()),
               str(out / "requests.txt"), str(context), str(out / "host.json"), str(prefill_rows)]
    save(out/'command.json', {'command': command, 'status': 'PENDING'})
    with (out / "stdout.jsonl").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr)
    save(out/'command.json', {'command': command, 'returncode': completed.returncode})
    if digest(binary) != build_info['binary_sha256']:
        raise ValueError('binary changed during measurement')
    if policy_hashes() != environment['environment_policy_hashes']:
        raise ValueError('environment policy changed during measurement')
    if any(digest(HERE/p['ids_file']) != p['sha256'] for p in prompts):
        raise ValueError('prompt IDs changed during measurement')
    if completed.returncode:
        raise ValueError(f"resident run failed ({completed.returncode}); inspect {out}/stderr.log")
    events = [json.loads(line) for line in (out / "stdout.jsonl").read_text().splitlines()]
    ready = [e for e in events if e["event"] == "ready"]
    if len(ready) != 1:
        raise ValueError("resident run must report exactly one checkpoint load")
    rows = [e for e in events if e["event"] == "result" and e["repeat"] >= 0]
    check_rows(rows, expected)
    by_name = {p["name"]: p for p in prompts}
    for row in rows:
        tokens = getattr(args, 'request_tokens', {}).get(row['name'], args.tokens)
        validate_result(row, by_name[row["name"]], tokens)
    host=json.loads((out/'host.json').read_text())
    if host.get('environment_policy_hashes') != environment['environment_policy_hashes']:
        raise ValueError('worker environment policy differs from the parent snapshot')
    serial_ceiling = args.serial_ceiling
    if serial_ceiling is None and args.checkpoint.name == corpus["checkpoint_name"]:
        serial_ceiling = SERIAL_305
    report = {"schema": 1, "build": build_info, "host": host,
              "queued_command": command, "queue_priority": priority,
              "environment_policy_hashes": environment['environment_policy_hashes'],
              "runner_sha256": RUNNER_SHA256, "rooflines_sha256": digest(HERE / "rooflines.py"),
              "corpus_sha256": digest(HERE / "manifest.json"),
              "checkpoint": str(args.checkpoint.resolve()), "tokenizer_sha256": corpus["tokenizer_sha256"],
              "config_sha256": digest(args.checkpoint / "config.json"),
              "quantization_sha256": digest(args.checkpoint / "quantization_config.json"), "load": ready[0],
              "prefill_rows": prefill_rows,
              "settings": {"mode": getattr(args, "mode", "decode"), "tokens": args.tokens, "chunk": args.chunk, "spec": args.spec,
                           "warmup": args.warmup, "repeat": args.repeat,
                           **({'request_tokens': args.request_tokens,
                               'request_warmup': args.request_warmup,
                               'prefill_names': list(args.prefill_names)} if hasattr(args, 'request_tokens') else {})},
              "ceilings": {"dram_gbs": 245.5, "wmma_f16_tflops": 49, "serial_tps": serial_ceiling,
                           "prefill_tps": args.prefill_ceiling, "spec_tps": args.spec_ceiling},
              "rows": [{**metrics(r, serial_ceiling, args.prefill_ceiling, args.spec_ceiling),
                        "prompt_sha256": by_name[r["name"]]["sha256"]} for r in rows]}
    report = annotate(report)
    save(out / "results.json", report)
    (out / "results.md").write_text(table(report))
    return report


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return result


def run(args):
    if "CRUCIBLE_PROFILE" in os.environ:
        raise ValueError("unset CRUCIBLE_PROFILE for suite measurements")
    prefill_only = getattr(args, "mode", "decode") == "prefill"
    if args.tokens < (1 if prefill_only else 2) or (prefill_only and args.tokens != 1) or \
            args.repeat < 1 or args.warmup not in (0, 1) or args.spec < 1 or args.chunk < 1:
        raise ValueError("use tokens >= 2 for decode or tokens=1 for prefill, repeat >= 1, warmup 0 or 1, positive spec/chunk")
    corpus = load_corpus(HERE / "manifest.json")
    if digest(args.checkpoint / "tokenizer.json") != corpus["tokenizer_sha256"]:
        raise ValueError("checkpoint tokenizer differs from the fixed corpus")
    selected = [p for p in corpus["prompts"] if p["name"] in args.prompts.split(",")]
    if {p["name"] for p in selected} != set(args.prompts.split(",")):
        raise ValueError("unknown prompt name")
    check_budget(selected, args.tokens, args.repeat, args.warmup, args.prefill_floor, 1 if prefill_only else 2)
    out = args.output.resolve()
    a = run_binary(args.binary, out / "a", selected, corpus, args, args.prefill_rows)
    if args.compare_binary:
        b = run_binary(args.compare_binary, out / "b", selected, corpus, args, args.compare_prefill_rows)
        save_comparison(out / "comparison.json", a, b)
    print(table(a))


def add_serve_parser(commands):
    serve = commands.add_parser("serve", help="measure per-turn wall TTFT through the resident engine protocol")
    serve.add_argument("checkpoint", type=Path)
    serve.add_argument("--engine", type=Path, required=True)
    serve.add_argument("--source-commit", required=True)
    serve.add_argument("--fixture", choices=("4k", "16k", "30k"), default="4k")
    serve.add_argument("--cache-control", choices=("auto", "bypass", "refresh"), default="auto")
    serve.add_argument("--legacy-protocol", action="store_true")
    serve.add_argument("--output", type=Path, required=True)
    serve.add_argument("--prefill-floor", type=positive, default=77)
    serve.add_argument("--gpu-run", dest="gpu_run", type=Path, default=work_paths.gpu_run())
    serve.add_argument("--gpu-wait", type=positive, default=3600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compile_parser = commands.add_parser("build", help="build a harness against a source worktree")
    compile_parser.add_argument("--source", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--hipcc", default="/opt/rocm/bin/hipcc")
    compile_parser.add_argument("--source-commit", help="commit for a git archive export")
    compile_parser.add_argument("--source-branch", help="branch for a git archive export")
    compile_parser.add_argument("--bulk-prefill", action="store_true", help="compile the explicit bulk prefill API adapter")
    run_parser = commands.add_parser("run", help="run one or two binaries under separate GPU locks")
    run_parser.add_argument("checkpoint", type=Path)
    run_parser.add_argument("--binary", type=Path, required=True)
    run_parser.add_argument("--compare-binary", type=Path)
    run_parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    run_parser.add_argument("--prefill-rows", type=int, default=0)
    run_parser.add_argument("--compare-prefill-rows", type=int, default=0)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--prompts", default=DEFAULT_PROMPTS)
    for flag, default in (("tokens", 160), ("repeat", 1), ("warmup", 1), ("spec", 2), ("chunk", 8)):
        run_parser.add_argument("--" + flag, type=int, default=default)
    run_parser.add_argument("--prefill-floor", type=positive, default=100)
    run_parser.add_argument("--priority", type=int, choices=range(4), default=3)
    run_parser.add_argument("--gpu-run", dest="gpu_run", type=Path, default=work_paths.gpu_run())
    run_parser.add_argument("--gpu-wait", type=positive, default=3600, help="queue wait seconds; GPU timeout stays 300 seconds")
    for name in ("serial", "prefill", "spec"):
        run_parser.add_argument(f"--{name}-ceiling", type=positive)
    add_serve_parser(commands)
    from regression import add_parser as add_regression_parser
    add_regression_parser(commands)
    annotation = commands.add_parser("annotate", help="add documented ceiling estimates to a saved report")
    annotation.add_argument("report", type=Path)
    annotation.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("compare", help="compare saved JSON reports without a GPU")
    comparison.add_argument("a", type=Path)
    comparison.add_argument("b", type=Path)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            build(args)
        elif args.command == "run":
            run(args)
        elif args.command == 'regression':
            from regression import launch
            launch(args)
        elif args.command == "serve":
            from conversation_run import launch
            launch(args)
        elif args.command == "annotate":
            report = {**annotate(json.loads(args.report.read_text())),
                      "annotation": {"source_report_sha256": digest(args.report),
                                     "rooflines_sha256": digest(HERE / "rooflines.py")}}
            save(args.output, report)
            args.output.with_suffix(".md").write_text(table(report))
        else:
            save_comparison(args.output, json.loads(args.a.read_text()), json.loads(args.b.read_text()))
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"suite: {error}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        worker()
    else:
        main()
