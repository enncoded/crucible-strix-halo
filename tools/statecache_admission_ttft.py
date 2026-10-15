#!/usr/bin/env python3
"""Measure stable-frontier misses, growing turns, and an explicit exact retry."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.suite.conversation_run import digest, load_turns
from tools import statecache_stream_ttft as streamed
from tools.statecache_ttft import event, validate_corpus


def verify(rows, expected):
    actual = {(r["case"], r["spec"], r["repeat"], r["phase"]): r for r in rows}
    if len(actual) != len(rows) or set(actual) != expected:
        raise RuntimeError("The admission capture has incomplete or duplicate cases.")
    for key, row in actual.items():
        case, spec, repeat, phase = key
        if any(not math.isfinite(row[k]) or row[k] <= 0 for k in ("ttft_seconds", "total_seconds")):
            raise RuntimeError("The admission capture has invalid durations.")
        if row["total_seconds"] < row["ttft_seconds"]:
            raise RuntimeError("The total duration is less than the first-token duration.")
        reference = "before" if case in ("4k", "exact4k", "16k_miss") else "bypass" + phase[-1]
        baseline = actual.get((case, spec, repeat, reference))
        if not baseline or row["output_ids"] != baseline["output_ids"] or row["sampling"] != baseline["sampling"]:
            raise RuntimeError("Cached and fresh complete IDs or sampling options differ.")
        if len(row["output_ids"]) != 32:
            raise RuntimeError("The admission capture needs 32 output IDs.")
        if phase not in ("retry", "auto1", "auto2") and (
                row["cached_tokens"] != 0 or row["prefill_tokens"] != row["input_tokens"]):
            raise RuntimeError("A bypass or refresh request reused cached target state.")
        for field in ("cache_saves", "cache_saved_bytes"):
            if type(row.get(field)) is not int or row[field] < 0:
                raise RuntimeError("The capture has no valid per-request snapshot counters.")
        expected_saves = int(phase == "miss" and case in ("16k_miss", "exact4k") or
                             phase == "auto0" and case in ("16k", "30k") or case == "30k" and phase == "auto1")
        if row["cache_saves"] != expected_saves or bool(row["cache_saved_bytes"]) != bool(expected_saves):
            raise RuntimeError("The request saved an absent or nonadmitted frontier.")
        if case == "4k" and any(row[k] for k in ("cached_tokens", "cache_bytes", "cache_saves", "cache_saved_bytes")):
            raise RuntimeError("The default short request saved or restored a snapshot.")
        required = case == "exact4k" and phase == "retry" or case in ("16k", "30k") and phase in ("auto1", "auto2")
        frontier = {"16k": 12288, "30k": 24576}.get(case)
        if required and (row["cached_tokens"] <= 0 or (frontier and row["cached_tokens"] < frontier)):
            raise RuntimeError("A growing turn or explicit exact retry restored no usable frontier.")
        if case == "exact4k" and phase == "retry" and row["prefill_tokens"]:
            raise RuntimeError("The explicit exact retry restored only a partial prompt.")
    for phase in ("before", "miss", "retry", "after"):
        serial = [r["output_ids"] for r in rows if r["case"] == "4k" and r["phase"] == phase]
        if serial and any(ids != serial[0] for ids in serial):
            raise RuntimeError("Serial and speculative default-short IDs differ.")


def summaries(rows, roofline):
    if not math.isfinite(roofline) or roofline <= 0:
        raise ValueError("Use a positive, finite prefill roofline.")
    groups = {}
    for row in rows:
        groups.setdefault((row["case"], row["spec"]), []).append(row)
    result = []
    for (case, spec), group in groups.items():
        values = []
        for repeat in sorted({r["repeat"] for r in group}):
            pair = {r["phase"]: r for r in group if r["repeat"] == repeat}
            before = pair.get("before", pair.get("bypass0"))
            after = pair.get("after", before)
            miss = pair.get("miss", pair.get("auto0"))
            reference = (before["ttft_seconds"] + after["ttft_seconds"]) / 2
            total = (before["total_seconds"] + after["total_seconds"]) / 2
            rate = miss["prefill_tokens"] / miss["ttft_seconds"]
            values.append(dict(miss_ttft_overhead_percent=100 * (miss["ttft_seconds"] / reference - 1),
                               miss_total_overhead_percent=100 * (miss["total_seconds"] / total - 1),
                               bypass_ttft_seconds=reference, miss_ttft_seconds=miss["ttft_seconds"],
                               bypass_total_seconds=total, miss_total_seconds=miss["total_seconds"],
                               miss_prefill_tps=rate, miss_roofline_percent=100 * rate / roofline))
        hits = [r for r in group if r["phase"] == "retry" or r["phase"] in ("auto1", "auto2")]
        hit_rows = []
        for hit in hits:
            phase = "before" if hit["phase"] == "retry" else "bypass" + hit["phase"][-1]
            baseline = next(r for r in group if r["repeat"] == hit["repeat"] and r["phase"] == phase)
            hit_rows.append(dict(phase=hit["phase"], cached_tokens=hit["cached_tokens"],
                                 ttft_seconds=hit["ttft_seconds"], total_seconds=hit["total_seconds"],
                                 ttft_speedup=baseline["ttft_seconds"] / hit["ttft_seconds"]))
        result.append(dict(case=case, spec=spec, repeats=len(values), aggregation="paired median",
                           retry_hit_rate=sum(r["cached_tokens"] > 0 for r in hits) / len(hits) if hits else None,
                           roofline_tps=roofline, hits=hit_rows,
                           **{k: statistics.median(v[k] for v in values) for k in values[0]}))
    return result


def checked_inputs(args, manifest, artifacts):
    streamed.verify_inputs(args, manifest, artifacts)
    if digest(Path(__file__)) != artifacts["admission_runner_sha256"]:
        raise RuntimeError("The frozen admission runner changed.")


def capture(args, requests, manifest, artifacts, exact_retry=False):
    rows = []
    expected = {(case, spec, repeat, phase) for case, spec, repeat, phase, *_ in requests}
    command = [str(args.engine), "serve", str(args.checkpoint), "--ctx", "32768", "--prefill", "4096",
               "--mtp-prefix", "256", "--conversation-cache-mib", "4096",
               "--conversation-cache-exact-retry", str(int(exact_retry))]
    report = dict(status="RUNNING", rows=rows, manifest=manifest, artifacts=artifacts,
                  command=command, exact_retry=exact_retry, expected=sorted(expected))
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    try:
        checked_inputs(args, manifest, artifacts)
        with args.output.with_suffix(".stderr.log").open("w") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=log, text=True, encoding="utf-8", bufsize=1)
            try:
                if event(process).get("event") != "ready":
                    raise RuntimeError("The engine did not return a ready event.")
                for case, spec, repeat, phase, ids, boundary, control in requests:
                    checked_inputs(args, manifest, artifacts)
                    row = dict(case=case, spec=spec, repeat=repeat, phase=phase, input_tokens=len(ids),
                               **streamed.stream_request(process, ids, boundary, control, spec, 32))
                    rows.append(row); save(); print(json.dumps(row), flush=True)
                    checked_inputs(args, manifest, artifacts)
            finally:
                process.stdin.close()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate(); process.wait(timeout=10)
            if process.returncode:
                raise RuntimeError(f"The engine exited with code {process.returncode}.")
        verify(rows, expected)
        report.update(status="PASS", summaries=summaries(rows, args.roofline_tps))
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine", "checkpoint", "fixtures", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--roofline-tps", type=float, default=2310)
    args = parser.parse_args()
    if not math.isfinite(args.roofline_tps) or args.roofline_tps <= 0:
        parser.error("--roofline-tps must be positive and finite")
    manifest_path = args.fixtures / "manifest.json"
    manifest_bytes = manifest_path.read_bytes(); manifest = json.loads(manifest_bytes)
    loaded = {name: tuple(load_turns(manifest_path, name, manifest=manifest)) for name in ("4k", "16k", "30k")}
    short = loaded["4k"][-1]
    requests = []
    for spec in (0, 2):
        for repeat in range(2):
            requests.extend(("4k", spec, repeat, phase, tuple(short["ids"]), short["cache_prefix_tokens"], control)
                            for phase, control in zip(("before", "miss", "retry", "after"), streamed.CONTROLS))
    for case in ("16k", "30k"):
        for turn, value in enumerate(loaded[case]):
            requests.extend((case, 2, 0, phase + str(turn), tuple(value["ids"]), value["cache_prefix_tokens"], control)
                            for phase, control in (("bypass", "bypass"), ("auto", "refresh" if turn == 0 else "auto")))
    full = loaded["16k"][-1]
    requests.extend(("16k_miss", 2, 0, phase, tuple(full["ids"]), full["cache_prefix_tokens"], control)
                    for phase, control in (("before", "bypass"), ("miss", "refresh")))
    proof = [("exact4k", 0, 0, phase, tuple(short["ids"]), short["cache_prefix_tokens"], control)
             for phase, control in zip(("before", "miss", "retry", "after"), streamed.CONTROLS)]
    artifacts = dict(engine_sha256=digest(args.engine), manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                     runner_sha256=digest(Path(streamed.__file__)), admission_runner_sha256=digest(Path(__file__)),
                     fixture_reader_sha256=digest(ROOT / "bench/suite/conversation_run.py"),
                     protocol_reader_sha256=digest(ROOT / "tools/statecache_ttft.py"),
                     checkpoint=validate_corpus(args.checkpoint, manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    streamed.reserve_outputs(args.output)
    capture(args, tuple(requests), manifest, artifacts)
    args.output = args.output.with_suffix(".proof.json")
    capture(args, tuple(proof), manifest, artifacts, exact_retry=True)


if __name__ == "__main__":
    main()
