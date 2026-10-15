#!/usr/bin/env python3
"""Check the serve-CLI equality run of results/v2/e2e.sh (one pack).

  python3 tools/vision_e2e_check.py DIR [--cases 4]

on.events:    images spec 0, text spec 0, images spec 7, text spec 7 (--vision 1, no cache)
off.events:   text spec 0, text spec 7 (vision off);  t12.events: the same on the train binary
cache.events: images spec 0 twice, then images spec 7 (--vision 1, conversation cache, session replay)
Every image reply must equal on.events images spec 0; every text reply must equal off.events text spec 0.
"""
import argparse
import hashlib
import json
import os


def results(path):
    out, ids = [], []
    for raw in open(path):
        event = json.loads(raw)
        if event.get("event") == "token":
            ids.append(event["id"])
        elif event.get("event") in ("done", "error"):
            out.append(dict(event=event, ids=ids))
            ids = []
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir")
    p.add_argument("--cases", type=int, default=4)
    args = p.parse_args()
    n = args.cases
    on, off, t12, cache = (results(os.path.join(args.dir, f"{name}.events"))
                           for name in ("on", "off", "t12", "cache"))
    errors = [r["event"] for r in on + off + t12 + cache if r["event"].get("event") == "error"]
    image, text = [r["ids"] for r in on[:n]], [r["ids"] for r in off[:n]]
    groups = {
        "image spec 7 (on)": [r["ids"] for r in on[2 * n:3 * n]],
        "image spec 0 cache pass 1": [r["ids"] for r in cache[:n]],
        "image spec 0 cache pass 2": [r["ids"] for r in cache[n:2 * n]],
        "image spec 7 cache": [r["ids"] for r in cache[2 * n:3 * n]],
    }
    text_groups = {
        "text spec 0 (on)": [r["ids"] for r in on[n:2 * n]],
        "text spec 7 (on)": [r["ids"] for r in on[3 * n:4 * n]],
        "text spec 7 (off)": [r["ids"] for r in off[n:2 * n]],
        "text spec 0 (t12)": [r["ids"] for r in t12[:n]],
        "text spec 7 (t12)": [r["ids"] for r in t12[n:2 * n]],
    }
    checks = {name: ids == image for name, ids in groups.items()}
    checks |= {name: ids == text for name, ids in text_groups.items()}
    cached = [r["event"].get("cached_tokens") for r in cache]
    digest = hashlib.sha256(json.dumps([image, text]).encode()).hexdigest()
    report = dict(dir=args.dir, errors=errors, checks=checks, cache_cached_tokens=cached,
                  vision_seconds=[r["event"].get("vision_seconds") for r in on[:n] + cache],
                  image_lengths=[len(i) for i in image], text_lengths=[len(t) for t in text], sha256=digest,
                  ok=not errors and all(checks.values()) and len(on) == 4 * n)
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
