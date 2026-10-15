#!/usr/bin/env python3
"""Batch serve: an image file that changes after the open ends only its request (review F1).

  python3 tools/vision_file_check.py ENGINE CKPT IMAGE_LINES.jsonl TEXT_LINES.jsonl OUT.json

Session A (text) decodes. Session B opens the first image request, its file then changes, and its next prefill must
return one error event for B (recoverable) while the engine keeps running. A then steps, and session C (the same image
request with an intact copy of the file) must give the same tokens as a fresh run of that request. Synthetic inputs.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile


def main():
    engine, ckpt, image_lines, text_lines, out = sys.argv[1:6]
    proc = subprocess.Popen([engine, "serve", ckpt, "--streams", "2", "--ctx", "16384", "--vision", "1"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    json.loads(proc.stdout.readline())

    def command(payload):
        proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        proc.stdin.flush()
        events = []
        while True:
            event = json.loads(proc.stdout.readline())
            if event.get("event") == "batch_end":
                return events
            events.append(event)

    def run(request_id, request):
        command(dict(op="open", request_id=request_id, request=request))
        tokens = []
        while True:
            events = command(dict(op="prefill", request_id=request_id) if not tokens else
                             dict(op="step", request_ids=[request_id]))
            tokens += [e["id"] for e in events if e.get("event") == "token"]
            if any(e.get("event") in ("done", "error") for e in events):
                return tokens, events

    text = json.loads(open(text_lines).readline())
    text.update(max_tokens=200, spec=0, stop_ids=[])
    image = json.loads(open(image_lines).readline())
    image.update(max_tokens=8, spec=0)
    work = tempfile.mkdtemp(prefix="vision-file-check-")
    source = image["images"][0]["path"]
    report = {}
    try:
        reference, _ = run(9, image)  # a fresh run of the image request
        command(dict(op="open", request_id=1, request=text))
        command(dict(op="prefill", request_id=1))
        changed = dict(image, images=[dict(image["images"][0], path=os.path.join(work, "changed.rgb"))])
        shutil.copyfile(source, changed["images"][0]["path"])
        opened = command(dict(op="open", request_id=2, request=changed))
        with open(changed["images"][0]["path"], "r+b") as file:
            file.write(b"\xff\x00\xff")  # other pixels: the digest no longer matches
        failed = command(dict(op="prefill", request_id=2))
        step = command(dict(op="step", request_ids=[1]))
        intact, _ = run(3, image)
        report = dict(opened=opened, failed=failed, peer_step=step, reference=reference, intact=intact,
                      engine_alive=proc.poll() is None)
        report["ok"] = (len(failed) == 1 and failed[0].get("event") == "error" and failed[0].get("request_id") == 2
                        and failed[0].get("recoverable") is True and "/" not in failed[0].get("error", "/")
                        and [e.get("event") for e in step] == ["token"] and intact == reference
                        and report["engine_alive"])
    finally:
        shutil.rmtree(work, ignore_errors=True)
        proc.stdin.close()
        proc.wait(timeout=60)
    json.dump(report, open(out, "w"), indent=1)
    print(json.dumps(dict(ok=report.get("ok"), failed=report.get("failed"))))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
