"""Synthetic and public prompts for the self-generated (on-policy) set.

  python -m training.mtp_tune.gen_prompts CKPT OUTDIR --root WORKTREE [--n 768] [--per 40]

Writes OUTDIR/ids/<name>.txt and OUTDIR/gen-<k>.cfg spec_sweep configs (production
decode policy, stop at <|im_end|>). Sources: UltraChat prompts that the d1 manifest
did not use (MIT), and synthetic code, tool and reasoning tasks written here. None
of them repeat the bench/tree_draft evaluation prompts.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from server.template import ChatTemplate  # noqa: E402

CODE_TASKS = [
    "a function that merges overlapping intervals", "a token bucket rate limiter class",
    "a function that parses ISO 8601 durations like P3DT4H5M", "a trie with insert, search and prefix listing",
    "a CLI that counts word frequencies in files given as arguments", "a binary heap priority queue",
    "a function that converts Roman numerals to integers and back", "a retry decorator with exponential backoff",
    "a thread-safe bounded queue", "a function that flattens nested lists of any depth",
    "a simple INI file parser", "Dijkstra's shortest path on an adjacency list",
    "a function that validates and normalizes email addresses", "a ring buffer with a fixed capacity",
    "a markdown table formatter for a list of rows", "a topological sort that reports cycles",
    "a function that computes the edit distance between two strings", "a small event emitter with on, off and emit",
    "a sliding window maximum over an array", "a JSON pretty printer without using a JSON library",
    "a union-find structure with path compression", "a function that splits a file into chunks of N lines",
    "an LFU cache", "a matrix multiplication with input checks", "a function that parses a cron expression",
    "a debounce helper", "a function that generates all permutations without recursion",
    "a bloom filter", "a URL query string parser", "a function that renders a progress bar in a terminal",
    "an expression evaluator for + - * / and parentheses", "a function that detects anagram groups in a word list",
    "a memoized Fibonacci with a cache size limit", "a simple key-value store with a write-ahead log",
    "a function that wraps text to a given width", "a k-means clustering step for 2D points",
    "a function that finds the median of two sorted arrays", "a semaphore-limited parallel downloader skeleton",
    "a function that compresses runs of characters (run-length encoding)", "a date range iterator",
]
LANGS = ["Python", "TypeScript", "Rust", "Go", "C++", "Python", "Java", "Python"]
STYLES = ["Add tests.", "Explain the design briefly, then give the code.", "Add type hints and docstrings.",
          "Keep it short.", "Handle invalid input with clear errors.", "Include a usage example."]
TOOL_TASKS = [
    "The build fails with 'undefined reference to compress_block'. Find where it is declared and fix the link.",
    "Rename the function get_user_data to load_user everywhere in src/ and update the tests.",
    "Add a --verbose flag to scripts/deploy.py that prints each step before it runs.",
    "Find all TODO comments in the repository and write them into TODO.md, grouped by file.",
    "The CI job says tests/test_parser.py::test_empty_input fails. Run it, find the cause, and fix the parser.",
    "Create a Makefile with build, test and clean targets for this C project in src/.",
    "Check which Python files import requests and replace it with urllib in the two smallest ones.",
    "Add a health check endpoint /healthz to app/server.py that returns JSON {\"ok\": true}.",
    "Count the lines of code per language in this repository and report the totals.",
    "The config loader ignores the HOME environment variable. Find the code and make it use it.",
    "Write a script that removes log files older than 14 days from logs/, then run it in dry-run mode.",
    "Upgrade the version string in setup.cfg and package/__init__.py to 2.4.0 and add a CHANGELOG entry.",
    "Find the slowest test in tests/ using pytest --durations and suggest a fix for it.",
    "Add input validation to the create_order function in shop/orders.py and add one test for it.",
    "List the git branches merged into main and delete the local ones that are older than 30 days.",
    "Make the logger in utils/log.py write to stderr instead of a file when LOG_FILE is unset.",
]
# Held-out set (--set held): tasks that do not appear in the train lists above.
CODE_TASKS_H = [
    "a function that checks whether a Sudoku board is valid", "an LRU cache with a time-to-live per entry",
    "a function that finds the longest common prefix of a list of strings", "a CSV to JSON converter CLI",
    "a function that computes a moving average over a stream", "a simple tokenizer for arithmetic expressions",
    "a function that detects a cycle in a linked list", "a rate-limited HTTP client wrapper",
    "a function that merges k sorted lists", "a minimal dependency injection container",
    "a function that converts a number to words in English", "a matrix transpose that works in place",
    "a function that balances parentheses by inserting the fewest characters", "a file watcher that polls mtimes",
    "a function that returns the n most common words in a text", "a circular doubly linked list",
    "a function that validates an IPv4 address", "a simple template engine with {{name}} placeholders",
    "a function that schedules meetings into the fewest rooms", "a bit set class with union and intersection",
]
TOOL_TASKS_H = [
    "The service crashes on startup with 'port already in use'. Find where the port is set and make it configurable.",
    "Add a unit test for the slugify function in utils/text.py and run the test suite.",
    "Replace all print calls in src/worker.py with logging calls at the right levels.",
    "Find out why tests/test_cache.py is flaky and make it deterministic.",
    "Write a script that checks every Markdown file in docs/ for broken relative links, then run it.",
    "Add type hints to every function in lib/geometry.py and run mypy on it.",
    "The Docker image is 1.2 GB. Look at the Dockerfile and reduce its size.",
    "Add a CHANGELOG.md entry for the bug fix in the last commit, using git log to find it.",
    "Move the constants in app/settings.py into a config.toml file and load them at startup.",
    "Find every function longer than 60 lines in src/ and list them with their lengths.",
]


def reasoning(rng):
    a, b, c = rng.randint(3, 40), rng.randint(2, 12), rng.randint(50, 500)
    return rng.choice([
        f"A train leaves at 09:{a:02d} and travels {c} km at {b * 10} km/h with one {a}-minute stop. When does it arrive?",
        f"How many integers between 1 and {c * 10} are divisible by {b} or {b + 1} but not by both?",
        f"A tank holds {c} liters. Pipe A fills it in {a} minutes, pipe B empties it in {a + b * 3} minutes. "
        "Both are open. How long until the tank is full?",
        f"Find all pairs of positive integers (x, y) with {b}x + {b + 3}y = {c}.",
        f"A rectangle has perimeter {c // 2 * 2} and its sides differ by {b}. Find its area.",
    ])


EDITS = ["Add docstrings to every public function in this module.",
         "Write unittest tests for the main functions of this module.",
         "Explain what this module does, then list three possible bugs or edge cases.",
         "Refactor the longest function in this module into smaller helpers. Return the full new code.",
         "Add type hints to this module. Return the complete updated file."]


def train2(rng, uc):
    """Second train set: more chat, combinatorial code tasks, edits of public library source, reasoning."""
    import glob
    items = [(f"t2chat{i:03d}", "chat", r["prompt"], i % 3 == 0, False) for i, r in enumerate(uc[85:175])]
    for i in range(70):
        t, lang, style = rng.choice(CODE_TASKS), rng.choice(LANGS), rng.choice(STYLES)
        items.append((f"t2code{i:03d}", "code", f"Write {t} in {lang}. {style}", i % 5 == 2, False))
    files = sorted(f for f in glob.glob("/usr/lib/python3*/*.py") if "test" not in f)
    rng.shuffle(files)
    for i, f in enumerate(files[:40]):
        src = open(f, encoding="utf-8", errors="replace").read()[:6000]
        name = f.rsplit("/", 1)[-1]
        items.append((f"t2edit{i:03d}", "code", f"{rng.choice(EDITS)}\n\n`{name}`:\n```python\n{src}\n```",
                      False, False))
    for i in range(24):
        items.append((f"t2think{i:03d}", "think", reasoning(rng), True, False))
    for i, t in enumerate(TOOL_TASKS):
        items.append((f"t2tool{i:03d}", "tool", t + " Explain each step before you call a tool.", False, True))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", default=os.path.expanduser("~/crucible-work/mtp-data/d1/manifest.json"))
    ap.add_argument("--ultrachat", default=os.path.expanduser("~/crucible-work/mtp-data/src/ultrachat_test_sft.parquet"))
    ap.add_argument("--n", type=int, default=768)
    ap.add_argument("--per", type=int, default=40, help="prompts per config (one lease)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--set", default="train", choices=["train", "held", "train2"])
    a = ap.parse_args()
    rng = random.Random(a.seed)
    from tokenizers import Tokenizer
    import pyarrow.parquet as pq
    tok = Tokenizer.from_file(str(a.ckpt / "tokenizer.json"))
    tmpl = ChatTemplate(a.ckpt)
    tools = json.loads(
        (ROOT / "training/mtp_tune/tools.json").read_text()) if (ROOT / "training/mtp_tune/tools.json").exists() else None
    used = set()
    for s in json.load(open(a.manifest))["shards"].values():
        used.update(s["keys"])
    uc = [r for r in pq.read_table(a.ultrachat, columns=["prompt_id", "prompt"]).to_pylist()
          if "uc:" + r["prompt_id"] not in used and len(r["prompt"]) < 3000]
    rng.shuffle(uc)
    held = a.set == "held"
    if held:
        # Same shuffle as the train set, then prompts after the 60 it used; new tasks and numbers.
        rng = random.Random(a.seed + 1000)
    pre = "h" if held else ""
    items = []
    if a.set == "train2":
        return write(a, tok, tmpl, tools, train2(rng, uc))
    for i, r in enumerate(uc[60:85] if held else uc[:60]):
        items.append((f"{pre}chat{i:03d}", "chat", r["prompt"], i % 3 == 0, False))
    for i, t in enumerate(CODE_TASKS_H if held else CODE_TASKS):
        lang = LANGS[(i + held) % len(LANGS)]
        items.append((f"{pre}code{i:03d}", "code", f"Write {t} in {lang}. {rng.choice(STYLES)}", i % 4 == 1, False))
    for i, t in enumerate(TOOL_TASKS_H if held else TOOL_TASKS):
        items.append((f"{pre}tool{i:03d}", "tool", t, False, tools is not None))
    for i in range(10 if held else 16):
        items.append((f"{pre}think{i:03d}", "think", reasoning(rng), True, False))
    return write(a, tok, tmpl, tools, items)


def write(a, tok, tmpl, tools, items):
    random.Random(a.seed).shuffle(items)
    (a.out / "ids").mkdir(parents=True, exist_ok=True)
    stop = tok.token_to_id("<|im_end|>")
    base = f"--n {a.n} --prefill 4096 --draft-head 64k --mtp-prefix 256 --stop-ids {stop} --spec 7 --draft-q 0.3"
    meta = []
    for k in range(0, len(items), a.per):
        lines = ["# generated by training/mtp_tune/gen_prompts.py"]
        for name, kind, text, think, use_tools in items[k:k + a.per]:
            msgs = [dict(role="user", content=text)]
            rendered = tmpl.render(msgs, enable_thinking=think, tools=tools if use_tools else ())
            ids = tok.encode(rendered, add_special_tokens=False).ids
            (a.out / "ids" / f"{name}.txt").write_text(",".join(map(str, ids)) + "\n")
            lines.append(f"{name} --ids-file {a.root}/ids/{name}.txt {base}")
            meta.append(dict(name=name, kind=kind, think=think, tools=use_tools, tokens=len(ids)))
        (a.out / f"gen-{k // a.per}.cfg").write_text("\n".join(lines) + "\n")
    (a.out / "prompts.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(len(items), "prompts,", (len(items) + a.per - 1) // a.per, "configs")


if __name__ == "__main__":
    main()
