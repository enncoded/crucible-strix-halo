#!/usr/bin/env python3
"""Build `crucible serve` JSON lines for image cases, and collect greedy IDs from its events.

  python tools/vision_probe.py lines CKPT CASES.json OUT.jsonl [--max-tokens N] [--spec S] [--text-only]
  python tools/vision_probe.py collect EVENTS.jsonl OUT.json

The request path is the HTTP server's (validate_request, the chat template, pad expansion),
without HTTP. --text-only writes only the case questions, for text-equality checks.
The engine reads pixels from files: `lines` writes them to OUT.jsonl.pixels/ with
the digest as the name. These are synthetic test images, so they stay for later runs.
"""
import argparse
import base64
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from server.api import STOP_IDS, validate_request  # noqa: E402
from server.tokenization import ModelTokenizer  # noqa: E402
from server.vision import VisionSettings, engine_images  # noqa: E402


def data_url(path):
    kind = "png" if path.lower().endswith(".png") else "jpeg"
    return f"data:image/{kind};base64," + base64.b64encode(open(path, "rb").read()).decode()


def pixel_paths(images, directory):
    os.makedirs(directory, mode=0o700, exist_ok=True)
    paths = []
    for image in images:
        path = os.path.abspath(os.path.join(directory, image.digest + ".rgb"))
        if not os.path.exists(path):
            with open(path, "wb") as file: file.write(image.rgb)
        paths.append(path)
    return paths


def lines(args):
    tokenizer = ModelTokenizer(args.ckpt)
    settings = VisionSettings()
    with open(args.out, "w") as out:
        for case in json.load(open(args.cases)):
            content = [dict(type="text", text=case["question"])]
            if not args.text_only:
                content = [dict(type="image_url", image_url=dict(url=data_url(os.path.join(os.path.dirname(args.cases), case["image"]))))] + content
            body = dict(messages=[dict(role="user", content=content)], max_tokens=args.max_tokens)
            request = validate_request(body, "chat", "crucible", vision=settings)
            ids, offsets = tokenizer.encode_images(request)
            line = dict(ids=ids, max_tokens=args.max_tokens, spec=args.spec, stop_ids=list(STOP_IDS))
            if request.images:
                line["images"] = engine_images(request.images, offsets, pixel_paths(request.images, args.out + ".pixels"))
            out.write(json.dumps(line, separators=(",", ":")) + "\n")
            print(case["name"], len(ids), [i.tokens for i in request.images], file=sys.stderr)


def collect(args):
    results, current = [], []
    for raw in open(args.events):
        event = json.loads(raw)
        if event.get("event") == "token":
            current.append(event["id"])
        elif event.get("event") in ("done", "error"):
            results.append(dict(event, ids=current))
            current = []
    json.dump(results, open(args.out, "w"), indent=1)
    for r in results:
        print(r.get("event"), r.get("prompt_tokens"), r.get("vision_seconds"), r.get("prefill_seconds"), r["ids"][:12],
              r.get("error", ""))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("lines")
    a.add_argument("ckpt"); a.add_argument("cases"); a.add_argument("out")
    a.add_argument("--max-tokens", type=int, default=32)
    a.add_argument("--spec", type=int, default=0)
    a.add_argument("--text-only", action="store_true")
    b = sub.add_parser("collect")
    b.add_argument("events"); b.add_argument("out")
    args = p.parse_args()
    (lines if args.command == "lines" else collect)(args)


if __name__ == "__main__":
    main()
