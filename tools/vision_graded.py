#!/usr/bin/env python3
"""Graded image questions (bench/vision_graded) against any OpenAI-compatible server, plus scoring.

Run against a server (Crucible or another OpenAI-compatible server):
  python3 tools/vision_graded.py run --base-url http://127.0.0.1:8796/v1 --model crucible \\
      --cases bench/vision_graded/set/cases.json --out results/graded-crucible.json
Score replies made elsewhere (ExLlamaV3: tools/vision_exllama_ref.py output):
  python3 tools/vision_graded.py score --cases .../cases.json --replies exllama.json --out graded-exl3.json

Only the Python standard library is needed. Requests use temperature 0 and thinking off.
"""
import argparse
import base64
import json
import os
import time
import urllib.request


def correct(reply, answers):
    text = reply.lower()
    for answer in answers:
        needed = answer if isinstance(answer, list) else [answer]
        if all(item.lower() in text for item in needed):
            return True
    return False


def data_url(path):
    kind = "jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "png"
    return f"data:image/{kind};base64," + base64.b64encode(open(path, "rb").read()).decode()


def ask(base_url, model, api_key, image, question, max_tokens, timeout, spec=None):
    body = dict(model=model, temperature=0, max_tokens=max_tokens, stream=False,
                chat_template_kwargs=dict(enable_thinking=False),
                messages=[dict(role="user", content=[dict(type="image_url", image_url=dict(url=image)),
                                                     dict(type="text", text=question)])])
    if spec is not None:  # Crucible only: batch mode needs spec 0
        body["spec"] = spec
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + api_key})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    return result, time.perf_counter() - start


def summary(rows):
    by = {}
    for row in rows:
        by.setdefault(row["category"], []).append(row["correct"])
    return dict(total=sum(r["correct"] for r in rows), cases=len(rows),
                by_category={k: f"{sum(v)}/{len(v)}" for k, v in sorted(by.items())})


def run(args):
    cases = json.load(open(args.cases))
    root = os.path.dirname(os.path.abspath(args.cases))
    rows = []
    for case in cases:
        try:
            result, seconds = ask(args.base_url, args.model, args.api_key, data_url(os.path.join(root, case["image"])),
                                  case["question"], args.max_tokens, args.timeout, args.spec)
            reply = result["choices"][0]["message"].get("content") or ""
            usage = result.get("usage", {})
        except Exception as error:  # a failed request scores zero and keeps its reason
            reply, seconds, usage = f"ERROR: {error}", 0.0, {}
        row = dict(name=case["name"], category=case["category"], question=case["question"], reply=reply,
                   correct=correct(reply, case["answers"]), seconds=round(seconds, 3),
                   prompt_tokens=usage.get("prompt_tokens"))
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(server=args.base_url, model=args.model, summary=summary(rows), rows=rows)
    json.dump(report, open(args.out, "w"), indent=1)
    print(json.dumps(report["summary"]))


def score(args):
    cases = {c["name"]: c for c in json.load(open(args.cases))}
    replies = {r["name"]: r.get("text", r.get("reply", "")) for r in json.load(open(args.replies))}
    rows = [dict(name=n, category=c["category"], question=c["question"], reply=replies.get(n, ""),
                 correct=correct(replies.get(n, ""), c["answers"])) for n, c in cases.items()]
    report = dict(source=args.replies, summary=summary(rows), rows=rows)
    json.dump(report, open(args.out, "w"), indent=1)
    print(json.dumps(report["summary"]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base-url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "local"))
    r.add_argument("--cases", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-tokens", type=int, default=128)
    r.add_argument("--timeout", type=float, default=300)
    r.add_argument("--spec", type=int, default=None, help="Crucible draft depth (send 0 to a --streams server)")
    s = sub.add_parser("score")
    s.add_argument("--cases", required=True)
    s.add_argument("--replies", required=True)
    s.add_argument("--out", required=True)
    args = p.parse_args()
    (run if args.command == "run" else score)(args)


if __name__ == "__main__":
    main()
