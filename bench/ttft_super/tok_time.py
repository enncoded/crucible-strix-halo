"""CPU timing of the server's per-request prompt work (render, encode, cache prefix, shared prefix).

Synthetic content only. Usage: python -I tok_time.py <repo> <checkpoint> <target_tokens>...
"""
import random
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, sys.argv[1])
from server.tokenization import ModelTokenizer  # noqa: E402

WORDS = ("alpha beta gamma delta engine kernel buffer stream token value index layer chunk cache state "
         "def return import class for while if else print list dict range open read write").split()


def messages(target_chars, rng):
    out = [dict(role="system", content="You are a coding agent. " * 200)]
    while sum(len(m["content"]) for m in out) < target_chars:
        body = " ".join(rng.choice(WORDS) for _ in range(800))
        out.append(dict(role="user", content=body))
        out.append(dict(role="assistant", content=body[::-1]))
    out.append(dict(role="user", content="next step"))
    return out


def request(msgs):
    return SimpleNamespace(kind="chat", messages=tuple(msgs), tools=(), enable_thinking=True, reasoning_effort="xhigh",
                           tool_choice=None, preserve_thinking=None, images=(), prompt=None)


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, 1000 * (time.perf_counter() - start)


def main():
    tok = ModelTokenizer(sys.argv[2])
    rng = random.Random(7)
    for target in map(int, sys.argv[3:]):
        req = request(messages(target * 5, rng))
        ids, t_enc = timed(lambda: tok.encode(req))
        _, t_render = timed(lambda: tok.render_prompt(req))
        _, t_cache = timed(lambda: tok.cache_prefix_tokens(req, ids))
        _, t_shared = timed(lambda: tok.shared_prefix_tokens(req, ids))
        batch_check(tok, req)
        print(f"tokens={len(ids)} encode_ms={t_enc:.1f} render_ms={t_render:.1f} "
              f"cache_prefix_ms={t_cache:.1f} shared_prefix_ms={t_shared:.1f}")



def batch_check(tok, req):
    """encode_batch of the prompt and its cache prefix: wall time and equality with two encodes."""
    text, _ = tok.render_prompt(req)
    prefix, _ = tok.render_prompt(req, add_generation_prompt=False)
    pair, t_pair = timed(lambda: tok.tokenizer.encode_batch([text, prefix], add_special_tokens=False))
    one, t_one = timed(lambda: [tok.tokenizer.encode(t, add_special_tokens=False) for t in (text, prefix)])
    same = [e.ids for e in pair] == [e.ids for e in one]
    print(f"tokens={len(one[0].ids)} batch_ms={t_pair:.1f} two_encodes_ms={t_one:.1f} equal={same}")


if __name__ == "__main__":
    main()
