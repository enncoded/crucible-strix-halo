#!/usr/bin/env python3
"""G15 A/B: one long agent-like session on serial serve, with or without --session-replay.

One model load per run. The replay arm builds the session live: each later prompt is the previous prompt, the
generated tokens and a new user chunk (a token-level client, so the template cannot drop tokens). The default
arm takes the replay arm's prompts (--prompts), so both arms prefill the same tokens and the outputs can be
compared per turn. Run under gpu-run with `timeout -k 10 300`.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.api import STOP_IDS, validate_request
from server.backend import EngineBackend

IM_END = 248046
ROOT = Path(__file__).resolve().parents[1]


def source_text(limit_chars):
    """Repository code as tool output: realistic for coding agents and the same on every host."""
    parts, total = [], 0
    # The layout move changes these paths and their order, so prompts from before it differ.
    for path in sorted((ROOT / "engine").rglob("*.h")) + sorted((ROOT / "server").glob("*.py")):
        text = f"=== {path.relative_to(ROOT)} ===\n{path.read_text()}\n"
        parts.append(text)
        total += len(text)
        if total >= limit_chars: break
    return "".join(parts)


def build_chunks(tokenizer, first, chunk, turns):
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    code = encode(source_text(8 * (first + chunk * turns)))
    if len(code) < first + chunk * turns: raise SystemExit("not enough source text for the session")
    head = encode("<|im_start|>system\nYou are a coding agent. Answer briefly.<|im_end|>\n<|im_start|>user\n"
                  "Read this code, then name one risk in it.\n")
    tail = encode("<|im_end|>\n<|im_start|>assistant\n")
    user = encode("\n<|im_start|>user\nTool result:\n")
    chunks = [head + code[:first] + tail]
    for turn in range(1, turns):
        start = first + chunk * (turn - 1)
        chunks.append(user + code[start:start + chunk] + encode("\nWhat else is risky?") + tail)
    return chunks


def session_chunks(turns):
    """The user chunks of an earlier run: each prompt minus the previous prompt and its generated tokens."""
    chunks = [turns[0]["prompt"]]
    for before, turn in zip(turns, turns[1:]):
        generated = before["ids"] + ([] if before["ids"] and before["ids"][-1] in STOP_IDS else [IM_END])
        head = before["prompt"] + generated
        if turn["prompt"][:len(head)] != head: raise SystemExit("the earlier run is not a token-level session")
        chunks.append(turn["prompt"][len(head):])
    return chunks


def run_turn(backend, ids, tokens, cache):
    body = dict(prompt=list(ids), max_tokens=tokens, cache_control=cache, temperature=0.0)
    request = validate_request(body, "completion", "crucible")
    started = time.monotonic()
    job = backend.submit(ids, request)
    out, first = [], None
    while True:
        event = job.events.get(timeout=280)
        if event["event"] == "error": raise RuntimeError(event)
        if event["event"] == "done": break
        if first is None: first = time.monotonic() - started
        out.append(event["id"])
    keys = ("cached_tokens", "prefill_tokens", "prefill_seconds", "decode_seconds", "session_replay",
            "cache_bytes", "cache_saves", "finish_reason", "cache_class")
    return dict(ids=out, ttft=first, seconds=time.monotonic() - started, prompt_tokens=len(ids),
                **{key: event.get(key) for key in keys})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--replay", type=int, choices=(0, 1), required=True)
    parser.add_argument("--prompts", help="replay-arm JSON whose prompts this run reuses")
    parser.add_argument("--session", help="earlier JSON whose user chunks this run reuses (same session, own outputs)")
    parser.add_argument("--first", type=int, default=30000)
    parser.add_argument("--chunk", type=int, default=1500)
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--gen", type=int, default=256)
    parser.add_argument("--cache-mib", type=int, default=8192)
    parser.add_argument("--ctx", type=int, default=65536)
    parser.add_argument("--bypass-last", action="store_true", help="also run the last prompt without the cache")
    parser.add_argument("--streams", type=int, help="batch serve with this many slots (default: serial serve)")
    parser.add_argument("--short-streams", type=int, default=0, help="batch slots of --short-ctx tokens (served layout)")
    parser.add_argument("--short-ctx", type=int, default=0)
    parser.add_argument("--prefill", type=int, default=None)
    options = parser.parse_args()
    if options.prompts:
        prompts = [turn["prompt"] for turn in json.loads(Path(options.prompts).read_text())["turns"]]
        chunks = None
    elif options.session:
        chunks, prompts = session_chunks(json.loads(Path(options.session).read_text())["turns"]), None
    else:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(Path(options.checkpoint) / "tokenizer.json"))
        chunks = build_chunks(tokenizer, options.first, options.chunk, options.turns)
        prompts = None
    loaded = time.monotonic()
    if options.streams:
        from server.batch_backend import BatchedEngineBackend
        backend = BatchedEngineBackend(options.engine, options.checkpoint, options.ctx, streams=options.streams,
                                       conversation_cache_mib=options.cache_mib, session_replay=bool(options.replay),
                                       short_streams=options.short_streams, short_ctx=options.short_ctx,
                                       prefill=options.prefill)
    else:
        backend = EngineBackend(options.engine, options.checkpoint, options.ctx, options.cache_mib,
                                engine_options=("--session-replay", "1") if options.replay else ())
    turns = []
    try:
        load_seconds = time.monotonic() - loaded
        ids = []
        for turn in range(len(prompts) if prompts else len(chunks)):
            if prompts: ids = prompts[turn]
            else:
                if turns:
                    generated = turns[-1]["ids"]
                    ids = ids + generated + ([] if generated and generated[-1] in STOP_IDS else [IM_END])
                ids = ids + chunks[turn]
            result = run_turn(backend, ids, options.gen, "auto")
            turns.append(dict(prompt=list(ids), **result))
            print(json.dumps({key: value for key, value in result.items() if key != "ids"}), flush=True)
        bypass = run_turn(backend, ids, options.gen, "bypass") if options.bypass_last else None
    finally:
        backend.close()
    report = dict(replay=options.replay, streams=options.streams, engine=options.engine, checkpoint=options.checkpoint,
                  load_seconds=load_seconds, turns=turns, bypass_last=bypass)
    Path(options.output).write_text(json.dumps(report))


if __name__ == "__main__":
    main()
