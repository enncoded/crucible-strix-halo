#!/usr/bin/env python3
"""Generate fixed IDs with the repository's tools/tok.py helper. CPU only."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SUITE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(checkpoint, text, chat=True):
    # In-process argv avoids the operating system's per-argument size limit.
    args = [str(ROOT / "tools/tok.py"), str(checkpoint), "encode", text]
    output = io.StringIO()
    with patch.object(sys, "argv", args + (["--chat"] if chat else [])):
        with contextlib.redirect_stdout(output):
            runpy.run_path(args[0], run_name="__main__")
    return [int(value) for value in output.getvalue().strip().split(",")]


def trim_middle(ids, length, tail_length):
    if length <= tail_length or len(ids) < length:
        raise ValueError("source must fill the target and retain its suffix")
    return ids[:length - tail_length] + ids[-tail_length:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    source = SUITE / "prompts.json"
    prompts = json.loads(source.read_text())
    rows = []
    for prompt in prompts["short"]:
        rows.append((prompt["name"], prompt["category"], encode(args.checkpoint, prompt["text"])))
    long = prompts["long"]
    tail_length = len(encode(args.checkpoint, long["suffix"])) + 16
    record_tokens = len(encode(args.checkpoint, long["record"], chat=False))
    for length in long["lengths"]:
        repeats = (length + record_tokens - 1) // record_tokens + 2
        text = long["prefix"] + long["record"] * repeats + long["suffix"]
        ids = trim_middle(encode(args.checkpoint, text), length, tail_length)
        rows.append((f"long_{length}", "long", ids))
    (SUITE / "ids").mkdir(exist_ok=True)
    entries = []
    for name, category, ids in rows:
        path = SUITE / "ids" / f"{name}.txt"
        path.write_text(",".join(map(str, ids)) + "\n")
        entries.append({"name": name, "category": category, "tokens": len(ids),
                        "ids_file": f"ids/{name}.txt", "sha256": sha256(path)})
    manifest = {"schema": 1, "checkpoint_name": args.checkpoint.name,
                "tokenizer_sha256": sha256(args.checkpoint / "tokenizer.json"),
                "source_sha256": sha256(source), "tok_py_sha256": sha256(ROOT / "tools/tok.py"),
                "chat_format": "tools/tok.py --chat, empty think block",
                "long_method": "Tokenize repeated records; trim the middle and retain the suffix.",
                "prompts": entries}
    (SUITE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
