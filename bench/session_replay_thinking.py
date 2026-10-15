#!/usr/bin/env python3
"""G15 thinking check over HTTP (127.0.0.1:8794, served batch layout): replay must hit after a thinking turn,
and a turn whose think block the server closed at the answer-room cap must replay or fall back with the result
of the fresh path. One lease: a replay-on server, then a replay-off server, the same two two-turn chats on each.
"""
import argparse
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.batch_backend import BatchedEngineBackend
from server.http import Service, handler_for
from server.tokenization import ModelTokenizer

PORT = 8794  # not 8790, the default port of a running server
ROOM = 200   # answer room: a 260-token request closes its think block after 60 tokens
CASES = (
    dict(name="think", max_tokens=900, first="Say hello in one short sentence.", second="Now say goodbye."),
    dict(name="capped", max_tokens=260, first="List the prime numbers below 60 and explain each check.",
         second="Which of them is largest?"),
)


def post(body):
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=200) as response:
        return json.loads(response.read())


def chat(messages, max_tokens):
    reply = post(dict(model="crucible", messages=messages, max_tokens=max_tokens, temperature=0, seed=7,
                      enable_thinking=True, return_token_ids=True, crucible_metrics=True))
    choice, usage = reply["choices"][0], reply.get("usage", {})
    metrics = usage.get("crucible_metrics") or {}
    return dict(token_ids=choice.get("token_ids"), message=choice["message"], finish=choice.get("finish_reason"),
                prompt_tokens=usage.get("prompt_tokens"), session_label=metrics.get("session_label"),
                cached_tokens=metrics.get("cached_tokens"), metrics=metrics)


def run_server(engine, checkpoint, replay):
    backend = BatchedEngineBackend(engine, checkpoint, 262144, streams=3, short_streams=2, short_ctx=131072,
                                   prefill=4096, conversation_cache_mib=10240, session_replay=bool(replay))
    server = ThreadingHTTPServer(("127.0.0.1", PORT), handler_for(Service(backend, ModelTokenizer(checkpoint),
                                                                          thinking_answer_room=ROOM)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    results = {}
    try:
        for case in CASES:
            messages = [dict(role="system", content="You are a helpful assistant."),
                        dict(role="user", content=case["first"])]
            first = chat(messages, case["max_tokens"])
            assistant = dict(role="assistant", content=first["message"].get("content") or "")
            if first["message"].get("reasoning_content"):
                assistant["reasoning_content"] = first["message"]["reasoning_content"]
            second = chat(messages + [assistant, dict(role="user", content=case["second"])], case["max_tokens"])
            results[case["name"]] = dict(first=first, second=second)
            print(json.dumps(dict(replay=replay, case=case["name"], first_label=first["session_label"],
                                  second_label=second["session_label"], second_cached=second["cached_tokens"],
                                  first_finish=first["finish"])), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        backend.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine")
    parser.add_argument("checkpoint")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    on = run_server(args.engine, args.checkpoint, 1)
    time.sleep(30)  # the first engine's memory must return before the second passes admission
    off = run_server(args.engine, args.checkpoint, 0)
    verdict = {}
    for case in CASES:
        name = case["name"]
        a, b = on[name], off[name]
        hit = a["second"]["session_label"] == "hit_replay"
        verdict[name] = dict(first_equal=a["first"]["token_ids"] == b["first"]["token_ids"], second_replay_hit=hit,
                             second_equal=a["second"]["token_ids"] == b["second"]["token_ids"],
                             ok=a["first"]["token_ids"] == b["first"]["token_ids"] and
                                (hit or a["second"]["token_ids"] == b["second"]["token_ids"]))
    args.output.write_text(json.dumps(dict(verdict=verdict, on=on, off=off), indent=1))
    print(json.dumps(verdict))
    return 0 if all(item["ok"] for item in verdict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
