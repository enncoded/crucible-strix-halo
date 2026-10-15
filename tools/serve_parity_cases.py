"""The fixed request set of tools/serve_parity.py (synthetic text only, one seed per sampled case).

A case is one HTTP request. `build` makes the body from the earlier results of the same server, so a replay turn
repeats the reply that this server gave (as a real client does). `gap` marks a request that the batch server
refuses today; the tool reports it as KNOWN_GAP, and as GAP_CLOSED when it works.
"""
import base64
from dataclasses import dataclass, field
import struct
import zlib

GROUPS = ("greedy", "sampled", "spec", "vision", "replay", "gap")
NAMES = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel")


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    body: dict = field(default_factory=dict)
    build: object = None  # callable(results: dict name -> result) -> body, for replay turns
    gap: bool = False
    needs_vision: bool = False
    needs_cache: bool = False
    # Rough load for the GPU estimate: prompt tokens (cold) and generated tokens.
    prompt_tokens: int = 60
    gen_tokens: int = 48

    def body_for(self, results):
        return self.build(results) if self.build else self.body


def chat(text, max_tokens=48, **extra):
    body = dict(messages=[dict(role="user", content=text)], max_tokens=max_tokens, temperature=0,
                return_token_ids=True, crucible_metrics=True)
    body.update(extra)
    return body


def long_prompt(rows):
    """About `rows` x 12 prompt tokens, long enough to cross the 4,096-row prefill chunk several times."""
    lines = [f"Record {index}: the {NAMES[index % 8]} unit reads {index * 37 % 1000} at station {index % 17}."
             for index in range(rows)]
    return "\n".join(lines) + "\nHow many records are above? Answer with one number."


def png_data_url(width=96, height=96):
    """A deterministic gradient PNG made with the standard library only (no image file in the repository)."""
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))
    rows = b"".join(b"\x00" + bytes(value for x in range(width)
                                    for value in ((x * 255 // width), (y * 255 // height), ((x + y) * 5) % 256))
                    for y in range(height))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
           chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def vision_body(question, max_tokens=40):
    content = [dict(type="image_url", image_url=dict(url=png_data_url())), dict(type="text", text=question)]
    return dict(messages=[dict(role="user", content=content)], max_tokens=max_tokens, temperature=0,
                return_token_ids=True, crucible_metrics=True)


def reply_text(result):
    return result.get("text") or ""


def second_turn(first, question, **extra):
    def build(results):
        turn = [dict(role="user", content=PROMPTS["story"]),
                dict(role="assistant", content=reply_text(results[first])),
                dict(role="user", content=question)]
        body = chat("", **extra)
        body["messages"] = turn
        return body
    return build


PROMPTS = dict(
    story="Write two short sentences about a lighthouse keeper who counts ships.",
    copy="Repeat this line three times, exactly: the quick brown fox jumps over the lazy dog near the old mill.",
)


def default_cases():
    """The request set. The order is fixed: replay turns follow their first turn."""
    cases = [
        Case("greedy-short", "greedy", chat(PROMPTS["story"], spec=0)),
        Case("greedy-one-token", "greedy", chat(PROMPTS["story"], max_tokens=1, spec=0), gen_tokens=1),
        Case("greedy-long-prompt", "greedy", chat(long_prompt(700), max_tokens=24, spec=0),
             prompt_tokens=9000, gen_tokens=24),
        Case("greedy-bypass", "greedy", chat(PROMPTS["story"], spec=0, cache_control="bypass")),
        Case("sampled-seed1", "sampled", chat(PROMPTS["story"], temperature=0.8, top_p=0.95, top_k=40, seed=1, spec=0)),
        Case("sampled-seed2", "sampled", chat(PROMPTS["story"], temperature=0.8, top_p=0.95, top_k=40, seed=2, spec=0)),
        Case("sampled-min-p", "sampled", chat(PROMPTS["story"], temperature=1.0, min_p=0.05, seed=3, spec=0)),
        Case("sampled-spec-seed1", "sampled",
             chat(PROMPTS["story"], temperature=0.8, top_p=0.95, top_k=40, seed=1, spec=7)),
        Case("spec-greedy", "spec", chat(PROMPTS["story"], max_tokens=96, spec=7), gen_tokens=96),
        Case("spec-copy-lookup", "spec", chat(PROMPTS["copy"], max_tokens=96, spec=7), gen_tokens=96),
        # the batched server accepts mtp_prefix 0, so this is a plain comparison now.
        Case("spec-mtp-prefix-0", "spec", chat(PROMPTS["story"], max_tokens=96, spec=7, mtp_prefix=0), gen_tokens=96),
        Case("spec-long-prompt", "spec", chat(long_prompt(300), max_tokens=64, spec=7),
             prompt_tokens=4000, gen_tokens=64),
        Case("vision-describe", "vision", vision_body("Name the main colors in this image."), needs_vision=True,
             prompt_tokens=200, gen_tokens=40),
        Case("vision-spec0", "vision", dict(vision_body("Name the main colors in this image."), spec=0),
             needs_vision=True, prompt_tokens=200, gen_tokens=40),
        Case("replay-t1", "replay", chat(PROMPTS["story"], max_tokens=64, spec=7), needs_cache=True, gen_tokens=64),
        Case("replay-t2", "replay", build=second_turn("replay-t1", "Now write one sentence about the sea.",
                                                      max_tokens=64, spec=7),
             needs_cache=True, prompt_tokens=140, gen_tokens=64),
        Case("replay-t2-spec0", "replay", build=second_turn("replay-t1", "Now write one sentence about the sea.",
                                                            max_tokens=64, spec=0),
             needs_cache=True, prompt_tokens=140, gen_tokens=64),
        Case("replay-t2-refresh", "replay", build=second_turn("replay-t1", "Now write one sentence about the sea.",
                                                              max_tokens=32, spec=0, cache_control="refresh"),
             needs_cache=True, prompt_tokens=140, gen_tokens=32),
        Case("gap-draft-head-full", "gap", chat(PROMPTS["story"], max_tokens=32, spec=7, draft_head="full"), gap=True),
        Case("gap-draft-adapt-1", "gap", chat(PROMPTS["story"], max_tokens=32, spec=7, draft_adapt=1), gap=True),
        Case("gap-draft-q-0", "gap", chat(PROMPTS["story"], max_tokens=32, spec=7, draft_q=0), gap=True),
    ]
    return tuple(cases)
