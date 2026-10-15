#!/usr/bin/env python3
"""Measure resident send-to-first-token time with scripted agent conversations."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.suite.conversation_run import digest, load_turns


def event(process):
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("The engine stopped. Check the stderr log.")
    value = json.loads(line)
    if value.get("event") == "error":
        raise RuntimeError(value["error"])
    return value


def request(process, ids, boundary, control):
    payload = json.dumps(dict(ids=ids, max_tokens=1, spec=0, stop_ids=[],
                              cache_prefix_tokens=boundary, cache_control=control), separators=(",", ":"))
    started = time.perf_counter()
    process.stdin.write(payload + "\n")
    process.stdin.flush()
    first = event(process)
    elapsed = time.perf_counter() - started
    if first.get("event") != "token" or type(first.get("id")) is not int or not 0 <= first["id"] < 248320:
        raise RuntimeError("The engine did not return a valid first token.")
    done = event(process)
    cached = done.get("cached_tokens")
    if (done.get("event") != "done" or
            any(type(done.get(key)) is not int for key in ("completion_tokens", "prompt_tokens", "prefill_tokens")) or
            done.get("completion_tokens") != 1 or done.get("prompt_tokens") != len(ids) or
            type(cached) is not int or not 0 <= cached <= len(ids) or
            done.get("prefill_tokens") != len(ids) - cached or done.get("cache_hit") is not (cached > 0)):
        raise RuntimeError("The engine did not return valid completion counts.")
    if control == "bypass" and cached:
        raise RuntimeError("The bypass request restored cached tokens.")
    return dict(ttft_seconds=elapsed, first_id=first["id"], **done)


def verify_rows(rows, manifest):
    expected = {(s["name"], t["turn"], mode) for s in manifest["scenarios"]
                for t in s["turns"] for mode in ("bypass", "auto")}
    actual = {(r["scenario"], r["turn"], r["mode"]): r for r in rows}
    if len(actual) != len(rows) or set(actual) != expected:
        raise RuntimeError("The timing run has missing, duplicate, or unexpected turns.")
    for scenario, turn, mode in expected:
        if mode == "auto" and actual[scenario, turn, mode]["first_id"] != actual[scenario, turn, "bypass"]["first_id"]:
            raise RuntimeError(f"The cached first token differs from bypass: {scenario}, turn {turn}.")


def summaries(rows, manifest, roofline):
    if not math.isfinite(roofline) or roofline <= 0:
        raise ValueError("The prefill roofline must be positive and finite.")
    verify_rows(rows, manifest)
    result = []
    for scenario in manifest["scenarios"]:
        name, last = scenario["name"], max(turn["turn"] for turn in scenario["turns"])
        off = next(r for r in rows if (r["scenario"], r["turn"], r["mode"]) == (name, last, "bypass"))
        on = next(r for r in rows if (r["scenario"], r["turn"], r["mode"]) == (name, last, "auto"))
        if any(not math.isfinite(r["ttft_seconds"]) or r["ttft_seconds"] <= 0 for r in (off, on)):
            raise RuntimeError("The first-token duration must be positive and finite.")
        rate = on["prefill_tokens"] / on["ttft_seconds"]
        result.append(dict(scenario=name, bypass_seconds=off["ttft_seconds"], cached_seconds=on["ttft_seconds"],
                           speedup=off["ttft_seconds"] / on["ttft_seconds"], cached_tokens=on["cached_tokens"],
                           residual_prefill_tps=rate, roofline_tps=roofline, roofline_percent=100 * rate / roofline))
    return result


def validate_corpus(checkpoint, manifest):
    template = manifest.get("template_file")
    if template not in ("chat_template.jinja", "tokenizer_config.json"):
        raise ValueError("The fixture must name a supported checkpoint template file.")
    tokenizer_hash = digest(checkpoint / "tokenizer.json")
    template_hash = digest(checkpoint / template)
    if tokenizer_hash != manifest.get("tokenizer_sha256"):
        raise ValueError("The conversation tokenizer differs from the checkpoint.")
    if template_hash != manifest.get("template_sha256"):
        raise ValueError("The conversation template differs from the checkpoint.")
    return dict(path=str(checkpoint.resolve()), tokenizer_sha256=tokenizer_hash, template_sha256=template_hash,
                metadata_sha256={name: digest(checkpoint / name) for name in
                                 ("config.json", "quantization_config.json", "model.safetensors.index.json")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--roofline-tps", type=float, default=2310)
    args = parser.parse_args()
    if not math.isfinite(args.roofline_tps) or args.roofline_tps <= 0:
        parser.error("--roofline-tps must be positive and finite")
    manifest_path = args.fixtures / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    # A timing pair must consume the same checked bytes, even if files change after launch.
    turns = tuple((scenario["name"], turn["turn"], tuple(turn["ids"]), turn["cache_prefix_tokens"])
                  for scenario in manifest["scenarios"]
                  for turn in load_turns(manifest_path, scenario["name"], manifest=manifest))
    artifacts = dict(engine_sha256=digest(args.engine), manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                     runner_sha256=digest(Path(__file__)),
                     fixture_reader_sha256=digest(ROOT / "bench/suite/conversation_run.py"),
                     checkpoint=validate_corpus(args.checkpoint, manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".stderr.log").open("w") as log:
        process = subprocess.Popen([str(args.engine), "serve", str(args.checkpoint), "--ctx", "32768",
                                    "--prefill", "4096", "--conversation-cache-mib", "4096"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                   text=True, encoding="utf-8", bufsize=1)
        rows = []
        try:
            if event(process).get("event") != "ready":
                raise RuntimeError("The engine did not return a ready event.")
            for control in ("bypass", "auto"):
                for scenario, turn, ids, boundary in turns:
                    row = dict(scenario=scenario, turn=turn, mode=control,
                               **request(process, ids, boundary, control))
                    rows.append(row)
                    args.output.write_text(json.dumps(dict(status="RUNNING", rows=rows, manifest=manifest,
                                                           artifacts=artifacts), indent=2) + "\n")
                    print(json.dumps(row), flush=True)
            report = summaries(rows, manifest, args.roofline_tps)
            args.output.write_text(json.dumps(dict(status="PASS", rows=rows, manifest=manifest,
                                                   artifacts=artifacts, summaries=report), indent=2) + "\n")
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
    for row in report:
        print(f"{row['scenario']}: {row['bypass_seconds']:.6f} -> {row['cached_seconds']:.6f} s, "
              f"{row['speedup']:.2f}x, saved {row['cached_tokens']} tokens, "
              f"{row['residual_prefill_tps']:.1f} residual tokens/s, "
              f"{row['roofline_percent']:.1f}% of {row['roofline_tps']:g} tokens/s")


if __name__ == "__main__":
    main()
