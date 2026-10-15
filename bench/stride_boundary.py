"""Supplemental boundary checks. Run inside gpu-run with a 300-second timeout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def ids(path):
    values = [int(value) for value in path.read_text().strip().split(",")]
    if not values or any(value < 0 or value >= 248320 for value in values):
        raise ValueError("The prompt has invalid token IDs.")
    return values


def capture(engine, checkpoint, large, small, output, padded):
    environment = {**os.environ, "CRUCIBLE_PF_PAD_H": "1" if padded else "0",
                   "HSA_USERPTR_FOR_PAGED_MEM": "0"}
    with (output / "stderr.log").open("w") as log:
        process = subprocess.Popen([str(engine), "serve", str(checkpoint)], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=log, text=True, env=environment)
        transcript, sequences = [], []

        def receive():
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("The resident engine stopped unexpectedly.")
            event = json.loads(line)
            transcript.append(event)
            return event

        try:
            if receive().get("event") != "ready":
                raise RuntimeError("The engine did not send a ready event.")
            for spec in (0, 2):
                for prompt in (large, small):
                    request = dict(ids=prompt, max_tokens=8, spec=spec, stop_ids=[])
                    process.stdin.write(json.dumps(request) + "\n")
                    process.stdin.flush()
                    tokens = []
                    while True:
                        event = receive()
                        if event.get("event") != "token":
                            break
                        tokens.append(event["id"])
                    if event.get("event") != "done" or len(tokens) != 8:
                        raise RuntimeError("The request did not return eight tokens.")
                    sequences.append(tokens)
            process.stdin.close()
            if process.wait(timeout=30) != 0:
                raise RuntimeError("The engine returned an error.")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            (output / "events.json").write_text(json.dumps(transcript, indent=2) + "\n")
    if sequences[:2] != sequences[2:]:
        raise ValueError("Serial and speculative continuations differ.")
    return sequences


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "baseline", "candidate", "large_ids", "small_ids", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    large, small = ids(args.large_ids), ids(args.small_ids)
    if len(large) != 4096 or len(small) != 24:
        raise ValueError("Use a 4K prompt and a 24-token prompt.")
    hashes = {name: digest(getattr(args, name)) for name in ("baseline", "candidate")}
    if hashes["baseline"] == hashes["candidate"]:
        raise ValueError("Use different baseline and candidate binaries.")
    args.output.mkdir()
    result = {"status": "FAIL", "binary_sha256": hashes, "candidate_pad_h": "1", "sequences": {}}
    try:
        for name in ("baseline", "candidate"):
            output = args.output / name
            output.mkdir()
            result["sequences"][name] = capture(getattr(args, name).resolve(), args.checkpoint,
                                                large, small, output, name == "candidate")
        if result["sequences"]["baseline"] != result["sequences"]["candidate"]:
            raise ValueError("The padded boundary changes continuation IDs.")
        result["status"] = "PASS"
    finally:
        (args.output / "boundary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: 4K serial/speculative requests and later small requests match the independent baseline.")


if __name__ == "__main__":
    main()
