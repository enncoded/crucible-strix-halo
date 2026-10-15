"""CPU-only serial and batch serve, with an opt-in chat-boundary split fault."""
import hashlib
import json
import os
import sys
import time


def option(name, default):
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


context = option("--ctx", 65536)
rows = option("--prefill", 4096)
streams = option("--streams", 1)
batched = "--streams" in sys.argv
cache_enabled = option("--conversation-cache-mib", 0) > 0
split = os.environ.get("FAKE_CONFIG_SPLIT") == "1"
# FAKE_CONFIG_PEER_STOP=N: a request that bypasses the cache (a gate peer) writes the stop ID as its Nth token, as
# the 3-ID peers did on crucible-t13 (EOS after 1 to 10 tokens, Oct 7). A prompt that ends in a count ("16, ") goes
# on counting instead, which is how the model treats the counting peers.
peer_stop = int(os.environ.get("FAKE_CONFIG_PEER_STOP", "0"))
STOP_ID, COMMA, SPACE, DIGITS = 248046, 11, 220, range(15, 25)
cache = set()
active = {}
generation = 0
ready = dict(event="ready", context=context)
if batched:
    ready |= dict(protocol="batch-v1", streams=streams, prefill_rows=rows, slot_contexts=[context] * streams)
print(json.dumps(ready), flush=True)


def state_for(request, slot=0):
    boundary = request.get("cache_prefix_tokens", 0)
    prefix = tuple(request["ids"][:boundary])
    cached = max((len(key) for key in cache if prefix[:len(key)] == key), default=0) if cache_enabled else 0
    semantic = {key: request.get(key) for key in ("ids", "seed", "temperature", "top_k", "top_p", "min_p", "spec")}
    base = int(hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()[:8], 16) % 50000 + 500
    return dict(request=request, slot=slot, generation=generation, count=0, cursor=cached,
                cached=cached if request.get("cache_control") == "auto" else 0, prefix=prefix, base=base)


def ends_early(request):
    ids = request["ids"]
    counting = len(ids) >= 3 and ids[-3] in DIGITS and ids[-2:] == [COMMA, SPACE]
    return peer_stop > 0 and request.get("cache_control") == "bypass" and not counting


def stops(state):
    return ends_early(state["request"]) and state["count"] == peer_stop


def finished(state):
    return stops(state) or state["count"] == state["request"]["max_tokens"]


def token_event(state, id=None):
    index = state["count"]
    token = state["base"] + index
    if split and not batched and cache_enabled and state["prefix"] and index == 3:
        token += 1
    if ends_early(state["request"]) and index + 1 == peer_stop:
        token = STOP_ID
    state["count"] += 1
    return dict(event="token", id=token) | (dict(request_id=id, slot_generation=state["generation"]) if id is not None else {})


def done_event(state, id=None):
    request = state["request"]
    if cache_enabled and request.get("cache_control") != "bypass" and len(state["prefix"]) >= max(rows, 1):
        cache.add(state["prefix"])
        cache.add(state["prefix"][:len(state["prefix"]) // max(rows, 1) * max(rows, 1)])
    event = dict(event="done", prompt_tokens=len(request["ids"]), completion_tokens=state["count"],
                 finish_reason="stop" if stops(state) else "length")
    if cache_enabled:
        event |= dict(cached_tokens=state["cached"], cache_class="session")
    return event | (dict(request_id=id, slot_generation=state["generation"]) if id is not None else {})


for line in sys.stdin:
    command = json.loads(line)
    time.sleep(0.001)
    if not batched:
        state = state_for(command)
        for _ in range(command["max_tokens"]):
            time.sleep(0.001)
            print(json.dumps(token_event(state)), flush=True)
            if stops(state):
                break
        print(json.dumps(done_event(state)), flush=True)
        continue
    op = command["op"]
    if op == "prefill" and command.get("request_ids"):
        # Interleaved prefill: one token per live decoder per pass, then the pass boundary (engine rule, chunk > 8).
        target = active[command["request_id"]]
        chunk = min(rows, len(target["request"]["ids"]) - target["cursor"]) if rows else 8
        every = command["every"]
        live = list(command["request_ids"])
        for layer in range(1, 47):
            if chunk <= 8 or not live or (layer + 1 + every // 2) % every: continue
            for id in list(live):
                print(json.dumps(token_event(active[id], id)), flush=True)
                if finished(active[id]):
                    print(json.dumps(done_event(active[id], id)), flush=True)
                    active.pop(id)
                    live.remove(id)
            print(json.dumps(dict(event="batch_part", op="verify_step")), flush=True)
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
        ids = command["request_ids"] if op in ("step", "verify_step") else [command["request_id"]]
        for id in ids:
            state = active[id]
            if op == "prefill" and rows and state["cursor"] + rows < len(state["request"]["ids"]):
                state["cursor"] += rows
                continue
            count = min(3, state["request"]["max_tokens"] - state["count"]) if op == "spec_step" else 1
            for _ in range(count):
                print(json.dumps(token_event(state, id)), flush=True)
                if stops(state):
                    break
            if finished(state):
                print(json.dumps(done_event(state, id)), flush=True)
                active.pop(id)
    print(json.dumps(dict(event="batch_end", op=op)), flush=True)
