#!/usr/bin/env python3
"""Build held-out chat prompts and frequency-ranked EXL3 vocabulary blocks.

Run on CPU with the checkpoint tokenizer. Corpus is repository source/docs plus
Python's standard library, not benchmark prompts or generated continuations.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

BLOCK = 128
VOCAB = 248320
PROMPTS = {
    "code_fibonacci": "Write a Python function that returns the first n Fibonacci numbers. Include a short explanation and tests.",
    "code_json": "Write a Python function that reads JSON Lines from a file, skips malformed lines, and counts records by their status field. Include tests using a temporary file.",
    "prose_science": "Explain how photosynthesis works to a curious twelve-year-old. Describe what happens to sunlight, water, carbon dioxide, and oxygen in clear paragraphs.",
    "prose_story": "Write the opening of a quiet short story about a retired train conductor returning to the seaside town where she grew up. Use concrete sensory details and no dialogue.",
    "chat_dinner": "I have chickpeas, rice, canned tomatoes, onions, and spinach. Suggest a simple dinner for two and give step-by-step cooking instructions. I have about half an hour.",
    "chat_advice": "My friend keeps cancelling our plans at the last minute. I want to explain how I feel without sounding accusatory. What could I say, and how should I approach the conversation?",
    "code_copy": "Repeat the following Python code exactly, inside one Python code block:\n\n" + "\n".join(
        f"def transform_{i}(value):\n    result = value * {i + 2}\n    return result + {i + 1}\n" for i in range(24)),
}


def rank_blocks(counts, size):
    if size <= 0 or size > VOCAB or size % BLOCK:
        raise ValueError("size must be a positive multiple of 128 within vocabulary")
    scores = Counter()
    for token, count in counts.items():
        if not 0 <= token < VOCAB or count < 0:
            raise ValueError("invalid token frequency")
        scores[token // BLOCK] += count
    ranked = sorted(range(VOCAB // BLOCK), key=lambda b: (-scores[b], b))
    return sorted(ranked[:size // BLOCK])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--corpus", type=Path, action="append")
    ap.add_argument("--chat-jsonl", type=Path, action="append", default=[])
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    root = Path(__file__).resolve().parents[1]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    for name, prompt in PROMPTS.items():
        result = subprocess.run([sys.executable, str(root / "tools/tok.py"),
                                 str(args.checkpoint), "encode", prompt, "--chat"],
                                text=True, capture_output=True, check=True)
        (out / (name + ".ids")).write_text(result.stdout)
    paths = args.corpus or [root, Path(sysconfig.get_path("stdlib"))]
    counts = Counter()
    files, nbytes = 0, 0
    for base in paths:
        for path in sorted(base.rglob("*")):
            if (path.suffix not in {".py", ".md", ".h", ".hip", ".cpp", ".txt"}
                    or any(p in {".venv", "site-packages", "build", ".git", "results"}
                           for p in path.parts)
                    or path.name.startswith("draft_") or not path.is_file()):
                continue
            if path.stat().st_size > 512 * 1024:
                continue
            text = path.read_text(errors="replace")
            counts.update(tok.encode(text, add_special_tokens=False).ids)
            files += 1
            nbytes += len(text.encode())
    for path in args.chat_jsonl:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            record = json.loads(line)
            for key in ("instruction", "context", "response"):
                text = record.get(key, "")
                if not isinstance(text, str):
                    raise ValueError(f"{path}:{number}: {key} must be text")
                counts.update(tok.encode(text, add_special_tokens=False).ids)
        files += 1
        nbytes += path.stat().st_size
    total = sum(counts.values())
    summary = {"files": files, "bytes": nbytes, "tokens": total,
               "distinct_tokens": len(counts), "subsets": {}}
    ranked_tokens = sorted(range(VOCAB), key=lambda t: (-counts[t], t))
    for size in (16384, 32768, 65536):
        tokens = sorted(ranked_tokens[:size])
        (out / f"tokens{size}.txt").write_text(",".join(map(str, tokens)) + "\n")
        blocks = rank_blocks(counts, size)
        chosen = set(blocks)
        coverage = sum(c for t, c in counts.items() if t // BLOCK in chosen) / total
        (out / f"blocks{size}.txt").write_text(",".join(map(str, blocks)) + "\n")
        summary["subsets"][size] = {"corpus_coverage": coverage, "blocks": len(blocks),
                                    "token_coverage": sum(counts[t] for t in tokens) / total}
    (out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
