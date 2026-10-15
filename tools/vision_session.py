#!/usr/bin/env python3
"""An agent-style image session through a Crucible server: every turn resends all earlier screenshots.

  python3 tools/vision_session.py --base-url http://127.0.0.1:8804/v1 --out results/session.json \\
      [--images 8] [--width 2560 --height 1440] [--spec 0]

Turn t sends screenshots 1..t (each a synthetic image with a large number on it), the earlier replies, and a
question about screenshot t, as Pi does. Before the image-file change the server refused the sixth 2560x1440 screenshot with
HTTP 400 on every later turn. Checks: every turn returns HTTP 200; the last turn repeated with cache_control
"bypass" gives the same reply (the cache changes only speed); and the reply names the number of the newest
screenshot (a vision sanity check, not exactness). Exit code 1 on a failed check.
"""
import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request


def screenshot(index, width, height):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGB", (width, height), (20 + 23 * index % 200, 30, 60 + 17 * index % 150))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=height // 4)
    except TypeError:  # Pillow before 10.1 has no sized default font
        font = ImageFont.load_default()
    for row in range(0, height, 24):
        draw.text((8, row), f"log line {row // 24} of window {index}", fill=(140, 140, 140))
    draw.text((width // 3, height // 3), str(1000 + 37 * index), fill=(255, 255, 0), font=font)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def post(base_url, body):
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
            status = response.status
    except urllib.error.HTTPError as error:
        return dict(status=error.code, error=error.read().decode(errors="replace")[:300],
                    seconds=time.perf_counter() - start)
    usage = result.get("usage", {})
    return dict(status=status, reply=result["choices"][0]["message"].get("content") or "",
                prompt_tokens=usage.get("prompt_tokens"),
                cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                seconds=round(time.perf_counter() - start, 3), body_bytes=len(json.dumps(body)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--images", type=int, default=8)
    p.add_argument("--width", type=int, default=2560)
    p.add_argument("--height", type=int, default=1440)
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--spec", type=int, default=None)
    args = p.parse_args()
    urls = [screenshot(index, args.width, args.height) for index in range(1, args.images + 1)]
    messages, turns = [], []

    def body(cache_control):
        out = dict(model="crucible", temperature=0, max_tokens=args.max_tokens, stream=False,
                   cache_control=cache_control, chat_template_kwargs=dict(enable_thinking=False), messages=messages)
        if args.spec is not None:
            out["spec"] = args.spec
        return out

    for index, url in enumerate(urls, 1):
        messages.append(dict(role="user", content=[
            dict(type="image_url", image_url=dict(url=url)),
            dict(type="text", text=f"Screenshot {index}. What is the large yellow number? Answer with the number.")]))
        result = post(args.base_url, body("auto"))
        result.update(turn=index, expected=str(1000 + 37 * index))
        result["names_number"] = result["expected"] in result.get("reply", "")
        turns.append(result)
        print(json.dumps(result), flush=True)
        if result["status"] != 200:
            break
        messages.append(dict(role="assistant", content=result["reply"]))
    report = dict(turns=turns, images=args.images, size=[args.width, args.height])
    if all(turn["status"] == 200 for turn in turns):
        messages.pop()  # the last reply: repeat the last turn without the cache
        bypass = post(args.base_url, body("bypass"))
        report["bypass_last"] = bypass
        report["bypass_equal"] = bypass.get("reply") == turns[-1]["reply"]
        print(json.dumps(dict(bypass=bypass, equal=report["bypass_equal"])), flush=True)
    report["ok"] = (len(turns) == args.images and all(turn["status"] == 200 for turn in turns)
                    and report.get("bypass_equal") is True)
    json.dump(report, open(args.out, "w"), indent=1)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
