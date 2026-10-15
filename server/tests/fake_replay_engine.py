"""CPU-only serial and batch serve with session replay tails, for the replay gate tests.

FAKE_REPLAY_FAULT=batch changes one token of every batch replay hit; FAKE_REPLAY_FAULT=nohit never replays.
As the real model, a request with a junk prompt of fewer than JUNK_PROMPT ids stops after its first token (the train 20
chain: the old 3-id peers gave 10, 1 and 1 tokens, so no bulk chunk had 2 live decoders).
"""
import hashlib
import json
import os
import sys
import time

JUNK_PROMPT = 8
IM_END = 248046


def option(name, default):
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


context = option("--ctx", 65536)
rows = option("--prefill", 4096)
streams = option("--streams", 1)
batched = "--streams" in sys.argv
replay = option("--session-replay", 0) == 1 and option("--conversation-cache-mib", 0) > 0
fault = os.environ.get("FAKE_REPLAY_FAULT", "")
tails = []
active = {}
generation = 0
ready = dict(event="ready", context=context)
if batched:
    ready |= dict(protocol="batch-v1", streams=streams, prefill_rows=rows, slot_contexts=[context] * streams)
print(json.dumps(ready), flush=True)


def state_for(request, slot=0):
    ids = request["ids"]
    hit = max((tail for tail in tails if len(tail) < len(ids) and ids[:len(tail)] == tail), key=len, default=None)
    if fault == "nohit" or request.get("cache_control") != "auto":
        hit = None
    semantic = {key: request.get(key) for key in ("ids", "seed", "temperature", "top_p")}
    base = int(hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()[:8], 16) % 50000 + 500
    return dict(request=request, slot=slot, generation=generation, out=[], hit=hit, base=base, cursor=0)


def limit(state):
    return 1 if len(state["request"]["ids"]) < JUNK_PROMPT else state["request"]["max_tokens"]


def token_event(state, id=None):
    token = state["base"] + len(state["out"]) if limit(state) > 1 else IM_END
    if fault == "batch" and batched and state["hit"] is not None and len(state["out"]) == 2:
        token += 1
    state["out"].append(token)
    return dict(event="token", id=token) | (dict(request_id=id, slot_generation=state["generation"]) if id is not None else {})


def done_event(state, id=None):
    request, out, hit = state["request"], state["out"], state["hit"]
    if replay and request.get("cache_control") != "bypass" and len(out) >= 2:
        tails.append(request["ids"] + out[:-1])
    event = dict(event="done", prompt_tokens=len(request["ids"]), completion_tokens=len(out),
                 finish_reason="stop" if out[-1:] == [IM_END] else "length",
                 cached_tokens=len(hit) if hit else 0)
    if hit:
        event |= dict(session_replay=True, cache_class="replay", replay_rows=1 if request.get("spec") else 0)
    return event | (dict(request_id=id, slot_generation=state["generation"]) if id is not None else {})


for line in sys.stdin:
    command = json.loads(line)
    time.sleep(0.001)
    if not batched:
        state = state_for(command)
        for _ in range(limit(state)):
            time.sleep(0.002)
            print(json.dumps(token_event(state)), flush=True)
        print(json.dumps(done_event(state)), flush=True)
        continue
    op = command["op"]
    if op == "open":
        generation += 1
        used = {state["slot"] for state in active.values()}
        slot = next(slot for slot in range(streams) if slot not in used)
        id = command["request_id"]
        active[id] = state_for(command["request"], slot)
        print(json.dumps(dict(event="opened", request_id=id, slot_generation=generation, slot_context=context)), flush=True)
    elif op == "close":
        active.pop(command["request_id"])
    else:
        # An interleaved prefill (request_ids and every): one verify pass of its decoders before the chunk's own events.
        if op == "prefill" and command.get("request_ids"):
            for id in command["request_ids"]:
                state = active[id]
                print(json.dumps(token_event(state, id)), flush=True)
                if len(state["out"]) == limit(state):
                    print(json.dumps(done_event(state, id)), flush=True)
                    active.pop(id)
            print(json.dumps(dict(event="batch_part", op="verify_step")), flush=True)
        ids = command["request_ids"] if op in ("step", "verify_step") else [command["request_id"]]
        for id in ids:
            state = active[id]
            if op == "prefill" and rows and state["cursor"] + rows < len(state["request"]["ids"]):
                state["cursor"] += rows
                continue
            time.sleep(0.002)
            count = min(3, limit(state) - len(state["out"])) if op == "spec_step" else 1
            for _ in range(count):
                print(json.dumps(token_event(state, id)), flush=True)
            if len(state["out"]) == limit(state):
                print(json.dumps(done_event(state, id)), flush=True)
                active.pop(id)
    print(json.dumps(dict(event="batch_end", op=op)), flush=True)
