#!/usr/bin/env python3
"""Restart check for the disk tier. Synthetic content only.

`turn1`: N sessions, each one long user message of about --tokens tokens (seeded synthetic lines), sent at once.
`turn2`: each session's next turn: the same message, the turn-1 answer, and a short new question, sent at once.
Each mode writes one JSON file with the answer text sha256, the client TTFT, and the server's prompt and cached
token counts. `compare` checks that two turn-2 files have the same answers. The answers are greedy (temperature 0),
thinking off.
"""
import argparse
import hashlib
import json
import random
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_soak import CHARS_PER_TOKEN, lines  # noqa: E402

SYSTEM = "You are a careful assistant. Answer in two short sentences."


def first_messages(session, tokens):
    text = lines(random.Random(1000 + session), tokens * CHARS_PER_TOKEN)
    return [dict(role="system", content=SYSTEM),
            dict(role="user", content=f"Session {session}. Read this listing.\n{text}\nName one file in it.")]


def ask(port, messages, max_tokens, bypass=False):
    body = dict(model="crucible", messages=messages, max_tokens=max_tokens, temperature=0, stream=True,
                stream_options=dict(include_usage=True), chat_template_kwargs=dict(enable_thinking=False))
    if bypass: body["cache_control"] = "bypass"
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    start, first, text, usage = time.monotonic(), None, [], None
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"): usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                piece = choice.get("delta", {}).get("content")
                if piece:
                    if first is None: first = time.monotonic()
                    text.append(piece)
    answer = "".join(text)
    details = (usage or {}).get("prompt_tokens_details") or {}
    return dict(answer=answer, sha256=hashlib.sha256(answer.encode()).hexdigest(),
                ttft_s=None if first is None else round(first - start, 3),
                prompt_tokens=(usage or {}).get("prompt_tokens"), cached_tokens=details.get("cached_tokens"))


def run_all(port, conversations, max_tokens, bypass=False):
    results = [None] * len(conversations)
    def one(index):
        results[index] = ask(port, conversations[index], max_tokens, bypass)
    threads = [threading.Thread(target=one, args=(index,)) for index in range(len(conversations))]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("turn1", "turn2", "compare"))
    parser.add_argument("--port", type=int, default=8806)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=100000)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--bypass", action="store_true", help="turn2 without any cache restore (a cold prefill)")
    parser.add_argument("--first", type=Path, help="turn2: the turn1 file")
    parser.add_argument("--out", type=Path)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()
    if args.mode == "compare":
        runs = [json.loads(path.read_text()) for path in args.files]
        same = all([r["sha256"] for r in run] == [r["sha256"] for r in runs[0]] for run in runs)
        print(json.dumps(dict(same=same, files=[str(path) for path in args.files])))
        return 0 if same else 1
    conversations = [first_messages(session, args.tokens) for session in range(args.sessions)]
    if args.mode == "turn2":
        first = json.loads(args.first.read_text())
        conversations = [messages + [dict(role="assistant", content=first[index]["answer"]),
                                     dict(role="user", content="Name one more file and its line number.")]
                         for index, messages in enumerate(conversations)]
    results = run_all(args.port, conversations, args.max_tokens, args.bypass)
    # The answers stay in this file for the compare step; the content is synthetic.
    args.out.write_text(json.dumps(results, indent=1))
    print(json.dumps([{key: r[key] for key in ("ttft_s", "prompt_tokens", "cached_tokens", "sha256")} for r in results]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
