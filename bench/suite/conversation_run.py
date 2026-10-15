"""Measure direct serve TTFT with fixed, teacher-forced conversation prefixes."""
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from server.backend import EngineBackend, Job


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_turns(manifest_path, name, *, manifest=None):
    manifest = json.loads(manifest_path.read_bytes()) if manifest is None else manifest
    scenario = next((s for s in manifest["scenarios"] if s["name"] == name), None)
    if scenario is None:
        raise ValueError("unknown conversation fixture")
    result = []
    for turn in scenario["turns"]:
        path = manifest_path.parent / turn["ids_file"]
        raw = manifest_path.parent / turn["messages_file"]
        if digest(raw) != turn["messages_sha256"]:
            raise ValueError("conversation messages hash differs")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != turn["ids_sha256"]:
            raise ValueError("conversation ID hash differs")
        ids = [int(value) for value in data.decode("utf-8").strip().split(",")]
        if len(ids) != turn["tokens"] or not ids or any(value < 0 or value >= 248320 for value in ids):
            raise ValueError("invalid conversation IDs")
        if not 0 < turn["cache_prefix_tokens"] <= len(ids):
            raise ValueError("invalid stable cache boundary")
        result.append({**turn, "ids": ids})
    return result


def request_for(turn, control, legacy):
    base = {"ids": turn["ids"], "max_tokens": 1, "spec": 0, "stop_ids": []}
    return base if legacy else {**base, "cache_control": control, "cache_prefix_tokens": turn["cache_prefix_tokens"]}


def measure_turn(backend, turn, control, legacy, events):
    request = request_for(turn, control, legacy)
    job = Job(request)
    start = time.monotonic()
    backend.jobs.put_nowait(job)
    first, tokens = None, []
    while True:
        try:
            event = job.events.get(timeout=300)
        except queue.Empty as error:
            raise ValueError("serve did not finish a request") from error
        elapsed = time.monotonic() - start
        events.write(json.dumps({"turn": turn["turn"], "elapsed_seconds": elapsed, "event": event}) + "\n")
        events.flush()
        if event["event"] == "error":
            raise ValueError(f"serve rejected turn {turn['turn']}: {event.get('error')}")
        if event["event"] == "token":
            if type(event.get("id")) is not int or not 0 <= event["id"] < 248320:
                raise ValueError("serve returned an invalid token")
            first = elapsed if first is None else first
            tokens.append(event["id"])
        elif event["event"] == "done":
            if event.get("prompt_tokens") != turn["tokens"]:
                raise ValueError("serve returned an unexpected prompt count")
            if len(tokens) != 1 or event.get("completion_tokens") != 1:
                raise ValueError("serve did not return the requested single token")
            return {"turn": turn["turn"], "prompt_tokens": turn["tokens"],
                    "cache_prefix_tokens": turn["cache_prefix_tokens"], "ttft_wall_seconds": first,
                    "request_wall_seconds": elapsed, "output_ids": tokens, "summary": event}


def measure(engine, checkpoint, turns, control, legacy, events_path):
    context = max(4096, max(turn["tokens"] for turn in turns) + 16)
    started = time.monotonic()
    backend = EngineBackend(engine, checkpoint, context)
    ready = time.monotonic() - started
    try:
        with events_path.open("w") as events:
            rows = [measure_turn(backend, turn, control, legacy, events) for turn in turns]
        return {"ready_seconds": ready, "context": context, "turns": rows}
    finally:
        backend.close()


def markdown(report):
    lines = [f"# Serve conversation: {report['fixture']}", "",
             "| Turn | Prompt IDs | Wall TTFT s | Cache hit | Cached IDs | Prefill IDs |",
             "|---|---:|---:|---|---:|---:|"]
    for row in report["turns"]:
        summary = row["summary"]
        lines.append(f"| {row['turn']} | {row['prompt_tokens']} | {row['ttft_wall_seconds']:.6f} | "
                     f"{summary.get('cache_hit', 'n/a')} | {summary.get('cached_tokens', 'n/a')} | "
                     f"{summary.get('prefill_tokens', 'n/a')} |")
    lines += ["", "Wall TTFT starts before request dispatch and ends at the first returned token.",
              "It excludes model load and GPU queue time. Histories are fixed, not generated from earlier timed responses."]
    return "\n".join(lines) + "\n"


def worker(config_path):
    config = json.loads(Path(config_path).read_text())
    os.environ.setdefault("HSA_USERPTR_FOR_PAGED_MEM", "0")
    manifest = Path(config["manifest"])
    turns = load_turns(manifest, config["fixture"])
    if digest(Path(config["engine"])) != config["engine_sha256"] or digest(manifest) != config["manifest_sha256"]:
        raise ValueError("engine or fixture changed while queued")
    out = Path(config["output"])
    host = {"meminfo": Path("/proc/meminfo").read_text(), "buddyinfo": Path("/proc/buddyinfo").read_text(),
            "HSA_USERPTR_FOR_PAGED_MEM": os.environ["HSA_USERPTR_FOR_PAGED_MEM"]}
    (out / "host.json").write_text(json.dumps(host, indent=2) + "\n")
    result = measure(Path(config["engine"]), Path(config["checkpoint"]), turns,
                     config["cache_control"], config["legacy_protocol"], out / "events.jsonl")
    report = {**config, **result, "host": host}
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "results.md").write_text(markdown(report))


def launch(args):
    manifest = HERE / "conversations/manifest.json"
    turns = load_turns(manifest, args.fixture)
    corpus = json.loads(manifest.read_text())
    if digest(args.checkpoint / "tokenizer.json") != corpus["tokenizer_sha256"]:
        raise ValueError("conversation tokenizer differs from checkpoint")
    if digest(args.checkpoint / corpus["template_file"]) != corpus["template_sha256"]:
        raise ValueError("conversation template differs from checkpoint")
    if args.legacy_protocol and args.cache_control != "bypass":
        raise ValueError("legacy protocol requires --cache-control bypass")
    # Admission uses no assumed cache hits. The hard process limit still applies.
    estimate = 35 + sum(turn["tokens"] for turn in turns) / args.prefill_floor
    if estimate > 270:
        raise ValueError("conversation exceeds the 270s estimate without cache hits; use a measured faster prefill floor")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = {"engine": str(args.engine.resolve()), "checkpoint": str(args.checkpoint.resolve()),
              "fixture": args.fixture, "manifest": str(manifest), "output": str(out),
              "cache_control": args.cache_control, "legacy_protocol": args.legacy_protocol,
              "source_commit": args.source_commit, "source_identity": "operator-supplied label",
              "engine_sha256": digest(args.engine), "manifest_sha256": digest(manifest),
              "checkpoint_metadata": {name: digest(args.checkpoint / name)
                                      for name in ("config.json", "quantization_config.json")},
              "tokenizer_sha256": corpus["tokenizer_sha256"],
              "runner_sha256": digest(HERE / "conversation_run.py"),
              "backend_sha256": digest(ROOT / "server/backend.py")}
    (out / "request.json").write_text(json.dumps(config, indent=2) + "\n")
    command = [str(args.gpu_run), "-p", "3", "-w", str(args.gpu_wait), "--", "timeout", "-k", "10", "300",
               sys.executable, str(HERE / "conversation_run.py"), str(out / "request.json")]
    with (out / "worker.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=log)
    (out / "exit.json").write_text(json.dumps({"returncode": result.returncode}) + "\n")
    if result.returncode:
        raise ValueError(f"serve measurement failed ({result.returncode}); inspect {out}/worker.log")
    if not (out / "results.json").exists():
        raise ValueError("serve worker returned no result")
    print((out / "results.md").read_text())


if __name__ == "__main__":
    worker(sys.argv[1])
