#!/usr/bin/env python3
"""Peer decode stall from one many-image request through a running batch server (train13-audit M2).

  python3 tools/vision_peer_stall.py --base-url http://127.0.0.1:8804/v1 --out RUN.json [--peers 3] [--images 5]

The peers stream long text replies. After each peer has its first token, one request sends --images different
2560x1440 screenshots with cache_control "bypass" (a cache miss: every image needs the encoder). The report gives each
peer's longest gap between two streamed chunks while the image request runs, the image request's time, and the
longest scheduler commands from /v1/crucible/trace. Synthetic text
and images only. Exit 0 when every request returned HTTP 200.
"""
import argparse
import base64
import io
import json
import sys
import threading
import time
import urllib.request


def screenshot(index):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (2560, 1440), (30 + 40 * index % 200, 40, 90))
    draw = ImageDraw.Draw(image)
    for row in range(0, 1440, 20):
        draw.text((10 + 7 * index, row), f"window {index} row {row // 20} " * 12, fill=(220, 220, 220))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def request(base_url, body, timeout=600):
    return urllib.request.urlopen(urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=timeout)


def peer(base_url, index, max_tokens, arrivals, first, errors):
    body = dict(model="crucible", temperature=0, max_tokens=max_tokens, stream=True, spec=0,
                chat_template_kwargs=dict(enable_thinking=False),
                messages=[dict(role="user", content=f"Peer {index}: write a long numbered list of {max_tokens} "
                                                    "short facts about rivers, one per line.")])
    try:
        with request(base_url, body) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if any((choice.get("delta") or {}).get("content") for choice in chunk.get("choices", [])):
                    arrivals.append(time.monotonic())
                    first.set()
    except Exception as error:  # reported, not raised: the main thread owns the verdict
        errors.append(f"peer {index}: {type(error).__name__}")
        first.set()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--peers", type=int, default=3)
    p.add_argument("--images", type=int, default=5)
    p.add_argument("--peer-tokens", type=int, default=2048)
    p.add_argument("--text-control", action="store_true",
                   help="then send a text-only prompt of the same length (the chunk gap without any image)")
    args = p.parse_args()
    urls = [screenshot(index) for index in range(args.images)]
    arrivals = [[] for _ in range(args.peers)]
    firsts = [threading.Event() for _ in range(args.peers)]
    errors = []
    threads = [threading.Thread(target=peer, args=(args.base_url, i, args.peer_tokens, arrivals[i], firsts[i], errors),
                                daemon=True) for i in range(args.peers)]
    for thread in threads:
        thread.start()
    for event in firsts:
        event.wait(timeout=300)
    time.sleep(2.0)  # every peer in steady decode
    body = dict(model="crucible", temperature=0, max_tokens=16, stream=False, spec=0, cache_control="bypass",
                chat_template_kwargs=dict(enable_thinking=False),
                messages=[dict(role="user", content=[dict(type="image_url", image_url=dict(url=url)) for url in urls] +
                               [dict(type="text", text="How many screenshots are there? Answer with a number.")])])
    def measure(body):
        t0 = time.monotonic()
        with request(args.base_url, body) as response:
            result = json.load(response)
            status = response.status
        t1 = time.monotonic()
        time.sleep(1.0)
        with urllib.request.urlopen(args.base_url.rstrip("/") + f"/crucible/trace?since={t0 - 1:.6f}", timeout=60) as r:
            trace = json.load(r)["data"]
        window = [entry for entry in trace if entry["t_end"] >= t0 and entry["t_start"] <= t1]
        longest = sorted(window, key=lambda entry: entry["t_end"] - entry["t_start"], reverse=True)[:5]

        def gaps(times):
            inside = [t for t in times if t0 - 0.5 <= t <= t1 + 0.5]
            return list(zip(inside, inside[1:]))

        def kind(a, b):
            # A gap is a chunk gap when a prefill command with prompt rows overlaps it; else the encoder phase.
            chunk = any(e["op"] == "prefill" and (e.get("rows") or 0) > 0 and e["t_start"] < b and e["t_end"] > a
                        for e in window)
            return "chunk" if chunk else "other"

        def longest_gap(pairs, want=None):
            values = [b - a for a, b in pairs if want is None or kind(a, b) == want]
            return round(max(values), 3) if values else None
        per_peer = [gaps(times) for times in arrivals]
        return dict(status=status, seconds=round(t1 - t0, 3), reply=result["choices"][0]["message"].get("content"),
                    prompt_tokens=result.get("usage", {}).get("prompt_tokens"),
                    peer_longest_gap_seconds=[longest_gap(pairs) for pairs in per_peer],
                    peer_longest_gap_in_chunks=[longest_gap(pairs, "chunk") for pairs in per_peer],
                    peer_longest_gap_outside_chunks=[longest_gap(pairs, "other") for pairs in per_peer],
                    peer_chunks_in_window=[sum(t0 <= t <= t1 for t in times) for times in arrivals],
                    longest_commands=[dict(op=e["op"], seconds=round(e["t_end"] - e["t_start"], 3), rows=e.get("rows"),
                                           engine_s=e.get("engine_s")) for e in longest],
                    open_seconds=[round(e["t_end"] - e["t_start"], 3) for e in window if e["op"] == "open"])
    image = measure(body)
    status = image["status"]
    report = dict(images=args.images, peers=args.peers, image_status=status, image_seconds=image["seconds"],
                  image_reply=image["reply"], prompt_tokens=image["prompt_tokens"],
                  peer_longest_gap_seconds=image["peer_longest_gap_seconds"],
                  peer_longest_gap_in_chunks=image["peer_longest_gap_in_chunks"],
                  peer_longest_gap_outside_chunks=image["peer_longest_gap_outside_chunks"],
                  peer_chunks_in_window=image["peer_chunks_in_window"], longest_commands=image["longest_commands"],
                  open_seconds=image["open_seconds"], errors=errors)
    if args.text_control:
        words = " ".join(f"item{i} lies on shelf {i % 97}." for i in range(image["prompt_tokens"] // 9))
        text = dict(body, messages=[dict(role="user", content=words + " How many shelves are there?")])
        report["text_control"] = measure(text)
        status = max(status, report["text_control"]["status"])
    json.dump(report, open(args.out, "w"), indent=1)
    print(json.dumps(report))
    for thread in threads:
        thread.join(timeout=600)
    return 0 if status == 200 and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
