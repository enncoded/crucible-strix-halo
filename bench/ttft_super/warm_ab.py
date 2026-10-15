"""Warm-turn TTFT client for the ttft-super ABAB. Synthetic content only; stdlib only.

One session: a cold base turn of about --base tokens, then --rounds rounds of warm appends (--appends tokens each).
Each warm turn sends the previous reply (content and reasoning_content), as Pi does. Odd turns are greedy, even turns
sampled (seeded). One JSON line per turn: client TTFT, the server's token counts and the output token IDs.
"""
import argparse
import json
import random
import time
import urllib.request

WORDS = ("alpha beta gamma delta engine kernel buffer stream token value index layer chunk cache state def return "
         "import class for while if else print list dict range open read write the a of to in is it on").split()
CHARS_PER_TOKEN = 4.8  # synthetic words: 7,682 tokens for 8,000 at 4.6 chars (checkpoint tokenizer)


def text(tokens, rng):
    words, size = [], 0
    while size < tokens * CHARS_PER_TOKEN:
        word = rng.choice(WORDS)
        words.append(word + ("\n" if rng.random() < 0.08 else " "))
        size += len(words[-1])
    return "".join(words)


def turn(url, model, messages, max_tokens, temperature, seed, timeout):
    body = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, seed=seed, stream=True,
                return_token_ids=True, stream_options=dict(include_usage=True))
    request = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    start, first, ids, content, reasoning, usage = time.perf_counter(), None, [], [], [], None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                if "token_id" in choice:
                    ids.append(choice["token_id"])
                    first = first or time.perf_counter()
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
    reply = dict(role="assistant", content="".join(content), reasoning_content="".join(reasoning))
    return dict(ttft_s=(first or time.perf_counter()) - start, ids=ids, usage=usage), reply


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True); p.add_argument("--model", required=True)
    p.add_argument("--arm", required=True); p.add_argument("--out", required=True)
    p.add_argument("--base", type=int, default=120000); p.add_argument("--appends", default="500,2000,8000")
    p.add_argument("--rounds", type=int, default=2); p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--seed", type=int, default=29); p.add_argument("--timeout", type=float, default=600)
    a = p.parse_args()
    rng = random.Random(a.seed)
    messages = [dict(role="system", content="You are a coding agent. Answer briefly."),
                dict(role="user", content=text(a.base, rng))]
    steps = [("cold", a.base)] + [(f"r{r}-{n}", n) for r in range(a.rounds) for n in map(int, a.appends.split(","))]
    with open(a.out, "a") as out:
        for index, (step, size) in enumerate(steps):
            if index:
                messages += [reply, dict(role="user", content=text(size, rng))]
            temperature = 0.0 if index % 2 else 0.7
            row, reply = turn(a.url, a.model, messages, a.max_tokens, temperature, a.seed + index, a.timeout)
            row.update(arm=a.arm, step=step, append=size, temperature=temperature, time=time.time())
            out.write(json.dumps(row) + "\n"); out.flush()
            print(f"{a.arm} {step} ttft={row['ttft_s']:.3f} ids={len(row['ids'])} usage={row['usage']}", flush=True)


if __name__ == "__main__":
    main()
