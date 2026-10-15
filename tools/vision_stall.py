#!/usr/bin/env python3
"""Batch-mode decode stall from image requests: the longest engine command of session B, during which session A
cannot decode.

  python tools/vision_stall.py ENGINE CKPT IMAGE_LINES.jsonl TEXT_LINES.jsonl OUT.json [--steps 40]

Session A (text) decodes with `step`. Session B then opens with each text and each image request and runs its
prefill commands; A takes one step between B's commands, as the server scheduler lets a decoder in between.
Each command is synchronous, so A waits for the whole command. Before the image-file change the encoder ran inside the open
command; with the image-file change the `opened` event names `vision_slices`, and that many prefill commands each run one
encoder slice. The report gives every B command time, the slice and chunk times, and the longest one.
"""
import json
import subprocess
import sys
import time


def main():
    engine, ckpt, image_lines, text_lines, out = sys.argv[1:6]
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 40
    proc = subprocess.Popen([engine, "serve", ckpt, "--streams", "2", "--ctx", "16384", "--vision", "1"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    ready = json.loads(proc.stdout.readline())

    def command(payload):
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        start = time.perf_counter()
        proc.stdin.write(line)
        proc.stdin.flush()
        events = []
        while True:
            event = json.loads(proc.stdout.readline())
            if event.get("event") == "batch_end":
                return time.perf_counter() - start, events
            events.append(event)

    text = json.loads(open(text_lines).readline())
    text.update(max_tokens=10000, spec=0, stop_ids=[])
    report = dict(ready=ready, cases=[])

    def step_a():
        seconds, events = command(dict(op="step", request_ids=[1]))
        if [e.get("event") for e in events] != ["token"]:
            raise RuntimeError(f"session A step failed: {events}")
        return seconds

    def open_and_prefill(request_id, request, interleave):
        seconds, events = command(dict(op="open", request_id=request_id, request=request))
        if any(e.get("event") == "error" for e in events):
            raise RuntimeError(f"open failed: {events}")
        slices = next((e.get("vision_slices", 0) for e in events if e.get("event") == "opened"), 0)
        commands = [("open", seconds)]
        while not any(e.get("event") in ("token", "done", "error") for e in events):
            if interleave:
                step_a()
            seconds, events = command(dict(op="prefill", request_id=request_id))
            kind = "slice" if len(commands) <= slices else "chunk"
            commands.append((kind, seconds))
        if not any(e.get("event") == "token" for e in events):
            raise RuntimeError(f"prefill failed: {events}")
        return slices, commands

    open_and_prefill(1, text, False)
    a_steps = [step_a() for _ in range(steps)]
    report["a_step_seconds_median"] = sorted(a_steps)[len(a_steps) // 2]
    next_id = 2
    # A text-only open is the baseline.
    lines = [("text", json.loads(raw)) for raw in open(text_lines)] + [("image", json.loads(raw)) for raw in open(image_lines)]
    for kind, request in lines:
        request.update(max_tokens=4, spec=0)
        slices, commands = open_and_prefill(next_id, request, True)
        command(dict(op="close", request_id=next_id))
        after = sorted(step_a() for _ in range(5))
        times = lambda name: [round(s, 4) for k, s in commands if k == name]  # noqa: E731
        case = dict(kind=kind, prompt_tokens=len(request["ids"]), images=len(request.get("images", [])),
                    vision_slices=slices, open_seconds=round(commands[0][1], 4), slice_seconds=times("slice"),
                    chunk_seconds=times("chunk"), longest_command_seconds=round(max(s for _, s in commands), 4),
                    longest_open_or_slice_seconds=round(max(s for k, s in commands if k != "chunk"), 4),
                    encoder_seconds=round(sum(times("slice")), 4), a_step_after_median=round(after[2], 4))
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
        next_id += 1
    proc.stdin.close()
    proc.wait(timeout=60)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
