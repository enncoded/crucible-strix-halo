#!/usr/bin/env python3
"""Image conversations through a Crucible server with a conversation cache: cache and replay exactness.

  python3 tools/vision_turns.py --base-url http://127.0.0.1:8793/v1 --cases bench/vision_graded/set/cases.json \\
      --out results/turns.json [--spec 0] [--names a,b]

For each case: turn 1 (image + question), then turn 2 (turn 1, its reply, a follow-up). Each turn runs
with cache_control "bypass" first, then twice with "auto" (a cold store, then a hit or a replay).
All three replies of a turn must be equal: the cache changes only speed. A text-only question
after the image turns must equal its reply before them. Exit code 1 on any difference.
--long-image PATH adds two questions on one large image after about 1200 tokens of system text, so a
prompt-state snapshot can end inside the image and the next request encodes only the image rest.
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision_graded import data_url  # noqa: E402

FOLLOW_UP = "Now explain your answer in one sentence."
TEXT = "Name three primary colors. Answer briefly."


def post(base_url, messages, cache_control, spec, max_tokens):
    body = dict(model="crucible", temperature=0, max_tokens=max_tokens, stream=False, cache_control=cache_control,
                chat_template_kwargs=dict(enable_thinking=False), messages=messages)
    if spec is not None:
        body["spec"] = spec
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    usage = result.get("usage", {})
    return dict(reply=result["choices"][0]["message"].get("content") or "", prompt_tokens=usage.get("prompt_tokens"),
                cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"))


def turn(args, messages):
    runs = [post(args.base_url, messages, mode, args.spec, args.max_tokens) for mode in ("bypass", "auto", "auto")]
    return dict(runs=runs, equal=len({r["reply"] for r in runs}) == 1)


def long_rows(args):
    system = " ".join(f"Note {i}: the inventory record lists item {i} on shelf {i % 7}." for i in range(100))
    image = dict(type="image_url", image_url=dict(url=data_url(args.long_image)))
    rows = []
    for question in ("What is the title text at the top of the image?", "Describe the image in one sentence."):
        result = turn(args, [dict(role="system", content=system), dict(role="user", content=[image, dict(type="text", text=question)])])
        rows.append(dict(name="long: " + question, turn1=result, turn2=dict(equal=True, runs=[])))
        print(json.dumps(dict(name=rows[-1]["name"], equal=result["equal"],
                              cached=[r["cached_tokens"] for r in result["runs"]],
                              prompt_tokens=result["runs"][0]["prompt_tokens"])), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--spec", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--names", default="")
    p.add_argument("--long-image", default="")
    args = p.parse_args()
    root = os.path.dirname(os.path.abspath(args.cases))
    cases = json.load(open(args.cases))
    if args.names:
        wanted = args.names.split(",")
        cases = [c for c in cases if c["name"] in wanted]
    text = [dict(role="user", content=TEXT)]
    text_before = post(args.base_url, text, "bypass", args.spec, args.max_tokens)
    rows = []
    for case in cases:
        first = [dict(role="user", content=[dict(type="image_url", image_url=dict(url=data_url(os.path.join(root, case["image"])))),
                                            dict(type="text", text=case["question"])])]
        one = turn(args, first)
        second = first + [dict(role="assistant", content=one["runs"][0]["reply"]), dict(role="user", content=FOLLOW_UP)]
        two = turn(args, second)
        rows.append(dict(name=case["name"], turn1=one, turn2=two))
        print(json.dumps(dict(name=case["name"], turn1_equal=one["equal"], turn2_equal=two["equal"],
                              cached=[r["cached_tokens"] for r in one["runs"] + two["runs"]])), flush=True)
    if args.long_image:
        rows += long_rows(args)
    text_after = post(args.base_url, text, "bypass", args.spec, args.max_tokens)
    ok = all(r["turn1"]["equal"] and r["turn2"]["equal"] for r in rows) and text_before["reply"] == text_after["reply"]
    report = dict(server=args.base_url, spec=args.spec, ok=ok, text_before=text_before, text_after=text_after, rows=rows)
    json.dump(report, open(args.out, "w"), indent=1)
    print(json.dumps(dict(ok=ok, cases=len(rows), text_equal=text_before["reply"] == text_after["reply"])))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
