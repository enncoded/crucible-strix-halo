#!/usr/bin/env python3
"""Build tool-use conversation fixtures with the checkpoint's actual chat template. CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a source file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run selected tests.",
     "parameters": {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]}}},
]
SYSTEM = "You are a coding assistant. Read the files before proposing changes. Preserve public interfaces. Use the supplied tools. Explain risks and tests."


def hashed(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def common_prefix(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


def call(name, arguments):
    return {"role": "assistant", "content": "I will inspect the relevant file.",
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]}


def code(records, stage):
    header = f"# Repository fixture, stage {stage}\nfrom collections import OrderedDict\n\n"
    functions = [f"def test_cache_case_{stage}_{i}(cache):\n"
                 f"    key = ('model_{i % 7}', {i})\n"
                 f"    cache.put(key, {{'value': {i}, 'version': {stage}}})\n"
                 f"    assert cache.get(key)['value'] == {i}\n\n" for i in range(records)]
    return header + "".join(functions)


def base_messages():
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Review this cache implementation for a coding service. Check key isolation and eviction behavior."},
            call("read_file", {"path": "src/cache.py"})]


def extension(previous, stage):
    return previous + [
        {"role": "assistant", "content": "The cache key must include the model identifier. I will add a focused regression test."},
        {"role": "user", "content": f"Review the next test group, iteration {stage}. Keep the existing public interface."},
        call("run_tests", {"target": f"tests/test_cache_group_{stage}.py"})]


def fill_turn(prefix, target, stage, encode):
    def candidate(count):
        return prefix + [{"role": "tool", "name": "read_file" if stage == 0 else "run_tests",
                          "content": code(count, stage)},
                         {"role": "user", "content": "Explain the next safe change. Start with one short sentence, then name the tests."}]
    low, high = 0, 1
    while len(encode(candidate(high))) < target:
        high *= 2
    while high - low > 1:
        middle = (low + high) // 2
        if len(encode(candidate(middle))) < target:
            low = middle
        else:
            high = middle
    options = [(candidate(n), encode(candidate(n))) for n in (low, high)]
    return min(options, key=lambda item: abs(len(item[1]) - target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "conversations")
    args = parser.parse_args()
    from server.template import ChatTemplate
    from tokenizers import Tokenizer
    template = ChatTemplate(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    def render(messages, generation=True):
        return template.template.render(messages=messages, tools=TOOLS, add_generation_prompt=generation,
                                        enable_thinking=False, reasoning_effort="xhigh", **template.special_tokens)
    def encode(messages):
        return tokenizer.encode(render(messages), add_special_tokens=False).ids
    scenarios = []
    for name, final in (("4k", 4096), ("16k", 16384), ("30k", 30720)):
        directory = args.output / name
        directory.mkdir(parents=True, exist_ok=True)
        turns, messages, previous_ids = [], [], []
        for stage, target in enumerate((final - 3000, final - 1500, final)):
            prefix = base_messages() if stage == 0 else extension(messages, stage)
            messages, ids = fill_turn(prefix, target, stage, encode)
            raw = directory / f"turn-{stage}.json"
            tokens = directory / f"turn-{stage}.ids.txt"
            raw.write_text(json.dumps({"messages": messages, "tools": TOOLS, "enable_thinking": False}, indent=2) + "\n")
            tokens.write_text(",".join(map(str, ids)) + "\n")
            shared = common_prefix(previous_ids, ids)
            cache_prefix = tokenizer.encode(render(messages, False), add_special_tokens=False).ids
            if ids[:len(cache_prefix)] != cache_prefix:
                raise ValueError("generation suffix changes the encoded cache prefix")
            turns.append({"cache_prefix_tokens": len(cache_prefix), "turn": stage, "messages_file": f"{name}/{raw.name}", "ids_file": f"{name}/{tokens.name}",
                          "tokens": len(ids), "target_tokens": target, "shared_prefix_tokens": shared,
                          "new_tokens": len(ids) - shared, "ids_sha256": hashed(tokens), "messages_sha256": hashed(raw)})
            previous_ids = ids
        scenarios.append({"name": name, "turns": turns})
    template_path = args.checkpoint / "chat_template.jinja"
    source = template_path if template_path.exists() else args.checkpoint / "tokenizer_config.json"
    manifest = {"schema": 1, "tokenizer_sha256": hashed(args.checkpoint / "tokenizer.json"),
                "template_sha256": hashed(source), "template_file": source.name,
                "method": "Actual checkpoint Jinja template with tools; complete code records nearest each token target.",
                "scripted_history": True, "scenarios": scenarios}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
