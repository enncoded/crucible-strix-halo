#!/usr/bin/env python3
"""Measure streamed miss and hit latency without changing the frozen one-token runner."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.suite.conversation_run import digest, load_turns
from tools.statecache_ttft import event, validate_corpus

PHASES = ("before", "miss", "hit", "after")
CONTROLS = ("bypass", "refresh", "auto", "bypass")


def stream_request(process, ids, boundary, control, spec, count, sampled=False):
    if type(count) is not int or not 2 <= count <= 160:
        raise ValueError("Use 2 to 160 streamed output tokens.")
    payload = dict(ids=ids, max_tokens=count, spec=spec, stop_ids=[],
                   cache_prefix_tokens=boundary, cache_control=control)
    if sampled:
        payload |= dict(temperature=0.7, top_k=32, top_p=0.95, seed=321)
    started = time.perf_counter()
    process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
    process.stdin.flush()
    tokens = []
    first_seconds = None
    while True:
        value = event(process)
        if value.get("event") == "done":
            done = value
            break
        token = value.get("id")
        if value.get("event") != "token" or type(token) is not int or not 0 <= token < 248320:
            raise RuntimeError("The engine returned an invalid streamed token.")
        tokens.append(token)
        if len(tokens) == 1:
            first_seconds = time.perf_counter() - started
        if len(tokens) > count:
            raise RuntimeError("The engine returned too many streamed tokens.")
    total_seconds = time.perf_counter() - started
    cached = done.get("cached_tokens")
    if (len(tokens) != count or
            any(type(done.get(k)) is not int for k in ("completion_tokens", "prompt_tokens", "prefill_tokens")) or
            done.get("completion_tokens") != count or done.get("prompt_tokens") != len(ids) or
            type(cached) is not int or not 0 <= cached <= len(ids) or
            done.get("prefill_tokens") != len(ids) - cached or done.get("cache_hit") is not (cached > 0) or
            (control != "auto" and cached)):
        raise RuntimeError("The engine returned invalid streamed completion counts.")
    if (first_seconds is None or not math.isfinite(first_seconds) or first_seconds <= 0 or
            not math.isfinite(total_seconds) or total_seconds < first_seconds):
        raise RuntimeError("The engine returned invalid streamed durations.")
    sampling = {key: payload[key] for key in ("temperature", "top_k", "top_p", "seed") if key in payload}
    return dict(ttft_seconds=first_seconds, total_seconds=total_seconds, output_ids=tokens, sampling=sampling, **done)


def summarize(rows, plan, roofline):
    expected = {(name, spec, repeat, phase) for name, spec, repeat in plan for phase in PHASES}
    actual = {(r["scenario"], r["spec"], r["repeat"], r["phase"]): r for r in rows}
    if len(rows) != len(actual) or set(actual) != expected:
        raise RuntimeError("The streamed timing run has incomplete or duplicate cases.")
    if not math.isfinite(roofline) or roofline <= 0:
        raise ValueError("Use a positive, finite prefill roofline.")
    if any(not math.isfinite(r[key]) or r[key] <= 0 for r in rows for key in ("ttft_seconds", "total_seconds")):
        raise RuntimeError("The streamed timing run has invalid durations.")
    groups = {}
    for name, spec, repeat in plan:
        pair = [actual[name, spec, repeat, phase] for phase in PHASES]
        before, miss, hit, after = pair
        if any(r["output_ids"] != before["output_ids"] for r in pair):
            raise RuntimeError("Cached and fresh streamed output IDs differ.")
        if any(r.get("sampling", {}) != before.get("sampling", {}) for r in pair):
            raise RuntimeError("Cached and fresh streamed sampling options differ.")
        hit_eligible = not (name == "4k" and spec == 2)
        if hit_eligible and hit["cached_tokens"] <= 0:
            raise RuntimeError("An eligible streamed hit restored no tokens.")
        if name.startswith("resume_") and hit["prefill_tokens"] != 0:
            raise RuntimeError("The prompt-end resume proof restored only a partial prefix.")
        reference = (before["ttft_seconds"] + after["ttft_seconds"]) / 2
        total_reference = (before["total_seconds"] + after["total_seconds"]) / 2
        rate = miss["prefill_tokens"] / miss["ttft_seconds"]
        groups.setdefault((name, spec), []).append(dict(
            bypass_ttft_seconds=reference, miss_ttft_seconds=miss["ttft_seconds"],
            miss_ttft_overhead_percent=100 * (miss["ttft_seconds"] / reference - 1),
            bypass_total_seconds=total_reference, miss_total_seconds=miss["total_seconds"],
            miss_total_overhead_percent=100 * (miss["total_seconds"] / total_reference - 1),
            hit_ttft_seconds=hit["ttft_seconds"], hit_total_seconds=hit["total_seconds"],
            hit_speedup=reference / hit["ttft_seconds"], hit_cached_tokens=hit["cached_tokens"],
            miss_prefill_tps=rate, miss_roofline_percent=100 * rate / roofline))
    # Sampling at different spec depths consumes a different RNG schedule.
    for name in {name for name, _ in groups if not name.endswith("_sampled")}:
        references = [r["output_ids"] for r in rows if r["scenario"] == name and r["phase"] == "before"]
        if any(ids != references[0] for ids in references):
            raise RuntimeError("Serial and speculative streamed output IDs differ.")
    return [dict(scenario=name, spec=spec, repeats=len(values), aggregation="paired median",
                 hit_eligible=not (name == "4k" and spec == 2),
                 hit_rate=sum(v["hit_cached_tokens"] > 0 for v in values) / len(values),
                 hit_limit="bounded MTP and bulk schedule" if name == "4k" and spec == 2 else None,
                 roofline_tps=roofline, **{key: statistics.median(v[key] for v in values) for key in values[0]})
            for (name, spec), values in groups.items()]


def capture(args, turns, plan, manifest, artifacts):
    rows = []
    with args.output.with_suffix(".stderr.log").open("w") as log:
        process = subprocess.Popen([str(args.engine), "serve", str(args.checkpoint), "--ctx", "32768",
                                    "--prefill", "4096", "--mtp-prefix", str(args.mtp_prefix), "--conversation-cache-mib", "4096"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                   text=True, encoding="utf-8", bufsize=1)
        try:
            if event(process).get("event") != "ready":
                raise RuntimeError("The engine did not return a ready event.")
            for name, spec, repeat in plan:
                ids, boundary = turns[name]
                for phase, control in zip(PHASES, CONTROLS):
                    row = dict(scenario=name, spec=spec, repeat=repeat, phase=phase,
                               **stream_request(process, ids, boundary, control, spec, args.tokens,
                                                sampled=name.endswith("_sampled")))
                    rows.append(row)
                    args.output.write_text(json.dumps(dict(status="RUNNING", rows=rows, manifest=manifest,
                                                           artifacts=artifacts), indent=2) + "\n")
                    print(json.dumps(row), flush=True)
            summaries = summarize(rows, plan, args.roofline_tps)
            args.output.write_text(json.dumps(dict(status="PASS", rows=rows, manifest=manifest,
                                                   artifacts=artifacts, summaries=summaries), indent=2) + "\n")
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            args.output.write_text(json.dumps(dict(status="FAIL", error=str(error), rows=rows, manifest=manifest,
                                                   artifacts=artifacts), indent=2) + "\n")
            raise
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
    return summaries


def reserve_outputs(output):
    proof = output.with_suffix(".proof.json")
    paths = (output, output.with_suffix(".stderr.log"), proof, proof.with_suffix(".stderr.log"))
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refuse to replace an existing capture: {path}")
    for path in paths:
        with path.open("x", encoding="utf-8") as stream:
            if path in (output, proof):
                stream.write('{"status":"PENDING"}\n')


def verify_inputs(args, manifest, artifacts):
    paths = dict(engine_sha256=args.engine, manifest_sha256=args.fixtures / "manifest.json",
                 runner_sha256=Path(__file__), fixture_reader_sha256=ROOT / "bench/suite/conversation_run.py",
                 protocol_reader_sha256=ROOT / "tools/statecache_ttft.py")
    for key, path in paths.items():
        if digest(path) != artifacts[key]:
            raise RuntimeError(f"A frozen capture input changed: {path}")
    if validate_corpus(args.checkpoint, manifest) != artifacts["checkpoint"]:
        raise RuntimeError("The frozen checkpoint identity changed.")


def checked_capture(args, turns, plan, manifest, artifacts):
    try:
        verify_inputs(args, manifest, artifacts)
        result = capture(args, turns, plan, manifest, artifacts)
        verify_inputs(args, manifest, artifacts)
        return result
    except (OSError, RuntimeError, ValueError) as error:
        report = json.loads(args.output.read_bytes())
        args.output.write_text(json.dumps(report | dict(status="FAIL", error=str(error)), indent=2) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser()
    for name in ("engine", "checkpoint", "fixtures", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--tokens", type=int, default=32, choices=range(2, 161))
    parser.add_argument("--short-repeats", type=int, default=3, choices=range(1, 4))
    parser.add_argument("--roofline-tps", type=float, default=2310)
    parser.add_argument("--mtp-prefix", type=int, default=256, choices=(0, 256))
    args = parser.parse_args()
    if not math.isfinite(args.roofline_tps) or args.roofline_tps <= 0:
        parser.error("--roofline-tps must be positive and finite")
    manifest_path = args.fixtures / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    # Freeze validated IDs before launch, then use the same tuple for every timing mode.
    loaded = {s["name"]: tuple(load_turns(manifest_path, s["name"], manifest=manifest)) for s in manifest["scenarios"]}
    turns = {name: (tuple(loaded[name][-1]["ids"]), loaded[name][-1]["cache_prefix_tokens"]) for name in ("4k", "16k")}
    plan = tuple((name, spec, repeat) for spec in (0, 2) for name in ("4k", "16k")
                 for repeat in range(args.short_repeats if name == "4k" else 1))
    artifacts = dict(engine_sha256=digest(args.engine), manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                     runner_sha256=digest(Path(__file__)), tokens=args.tokens, plan=plan, mtp_prefix=args.mtp_prefix,
                     fixture_reader_sha256=digest(ROOT / "bench/suite/conversation_run.py"),
                     protocol_reader_sha256=digest(ROOT / "tools/statecache_ttft.py"),
                     checkpoint=validate_corpus(args.checkpoint, manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reserve_outputs(args.output)
    for row in checked_capture(args, turns, plan, manifest, artifacts):
        print(json.dumps(row), flush=True)
    # A bounded warmup excludes the prompt-end snapshot. Prove its MTP resume separately without that cap.
    small = loaded["4k"][0]
    proof_turns = {name: (tuple(small["ids"]), small["cache_prefix_tokens"])
                   for name in ("resume_greedy", "resume_sampled")}
    proof_plan = tuple((name, spec, 0) for name in proof_turns for spec in (0, 2))
    proof_args = SimpleNamespace(**(vars(args) | dict(output=args.output.with_suffix(".proof.json"), mtp_prefix=0)))
    proof_artifacts = artifacts | dict(plan=proof_plan, mtp_prefix=0)
    for row in checked_capture(proof_args, proof_turns, proof_plan, manifest, proof_artifacts):
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
