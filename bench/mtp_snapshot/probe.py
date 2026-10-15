#!/usr/bin/env python3
"""mtp-snapshot serve probe, one batched engine process.

The full MTP window (mtp_prefix 0) must give the IDs of the default window (256) at an equal restore point, cold and
from the cache: a prompt frontier with MTP rows, a replay tail, and slot resume. Rule of Oct 10: two windows can
restore a replay hit at different points (a replay hit follows its own row schedule); such a pair is listed as not
compared. A group of two windows with no pair at an equal restore point is NO PROOF (exit 1), never PASS. Rows of one
window must all be equal. The restore point is cached_tokens only, not the resume kind (cache_class, slot_resume):
this compares more pairs, so it can give a false FAIL but never a false PASS. Greedy and sampled, alone and with concurrent sessions. The
probe also records the TTFT and the cache fields of each request; with CRUCIBLE_SPEC_STATS=1 the engine log has the
draft counts and whether the warm-up continued restored MTP rows (`check` reads them).

  run ENGINE CKPT OUT.json [--prompt-tokens N]   (inside one GPU lease, stderr to a log file)
  check OUT.json ENGINE.log                      (CPU; prints the verdict, exit 0 on PASS)
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from server.api import validate_request  # noqa: E402
from tools.serve_replay_gate import IM_END, STOP_IDS, collect, digest, synthetic_code  # noqa: E402

HEAD = "<|im_start|>system\nYou are a coding agent. Answer briefly.<|im_end|>\n<|im_start|>user\n"
ASK = "Read this code, then explain what handler_{} returns.\n"
TAIL = "<|im_end|>\n<|im_start|>assistant\n"
MODES = dict(greedy=dict(temperature=0.0, seed=1), sampled=dict(temperature=0.7, seed=11, top_p=0.95))


def prompt(encode, offset, tokens):
    code = encode(synthetic_code(offset, tokens // 10 + 64))[:tokens]
    return encode(HEAD + ASK.format(offset + 7)) + code + encode(TAIL)


def next_turn(encode, ids, output, offset):
    closed = output + ([] if output and output[-1] in STOP_IDS else [IM_END])
    chunk = encode("\n<|im_start|>user\nTool result:\n") + encode(synthetic_code(offset, 220))[:2000] + \
        encode("\nWhat else is risky?" + TAIL)
    return ids + closed + chunk


def request(ids, mode, mtp_prefix, cache, max_tokens):
    body = dict(prompt=list(ids), max_tokens=max_tokens, spec=7, mtp_prefix=mtp_prefix, cache_control=cache,
                **MODES[mode])
    return validate_request(body, "completion", "crucible")


def run(args):
    from tokenizers import Tokenizer
    from server.batch_backend import BatchedEngineBackend
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids  # noqa: E731
    backend = BatchedEngineBackend(args.engine, args.checkpoint, args.ctx, streams=4, prefill=4096,
                                   conversation_cache_mib=args.cache_mib, session_replay=True, prefill_interleave=10)
    rows = []
    def one(label, ids, mode, mtp_prefix, cache, group):
        row = collect(backend, ids, request(ids, mode, mtp_prefix, cache, args.max_tokens), 280)
        done = row["done"]
        rows.append(dict(label=label, group=group, mode=mode, mtp_prefix=mtp_prefix, cache=cache,
                         prompt_tokens=len(ids), prompt_sha256=digest(ids), output_ids=row["output_ids"],
                         ttft=row["ttft"], seconds=row["seconds"], request_id=done.get("request_id"),
                         cached_tokens=done.get("cached_tokens"), cache_class=done.get("cache_class"),
                         slot_resume=done.get("slot_resume", False), decode_seconds=done.get("decode_seconds"),
                         prefill_seconds=done.get("prefill_seconds")))
        print(f"{label}: {len(ids)} tokens cached {done.get('cached_tokens')} ttft {row['ttft']:.2f} s", flush=True)
        return row["output_ids"]
    try:
        for index, mode in enumerate(MODES):
            ids = prompt(encode, 1000 * (index + 1), args.prompt_tokens)
            group = f"{mode}-a"
            one(f"{group}-ref", ids, mode, 256, "bypass", group)
            one(f"{group}-cold", ids, mode, 0, "bypass", group)
            out = one(f"{group}-save", ids, mode, 0, "auto", group)
            one(f"{group}-hit", ids, mode, 0, "auto", group)
            turn = next_turn(encode, ids, out, 50000 + index)
            # A replay hit follows the session's own row schedule, so its output can differ from a cold run of the
            # same prompt (tools/serve_replay_gate.py). It must equal a replay hit with the default window.
            group = f"{mode}-b"
            one(f"{group}-hit", turn, mode, 0, "auto", group)
            one(f"{group}-hit256", turn, mode, 256, "auto", group)
            one(f"{mode}-c-cold", turn, mode, 0, "bypass", f"{mode}-c")
            one(f"{mode}-c-ref", turn, mode, 256, "bypass", f"{mode}-c")
        # Three sessions at once: cold, then their second turns (replay) at once, against solo references.
        firsts = [prompt(encode, 7000 + 300 * i, args.prompt_tokens // 3) for i in range(3)]
        refs = [one(f"conc{i}-t1-ref", ids, "greedy", 256, "bypass", f"conc{i}-t1") for i, ids in enumerate(firsts)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            outs = list(pool.map(lambda item: one(f"conc{item[0]}-t1-full", item[1], "greedy", 0, "auto",
                                                  f"conc{item[0]}-t1"), enumerate(firsts)))
        seconds = [next_turn(encode, ids, out, 60000 + i) for i, (ids, out) in enumerate(zip(firsts, outs))]
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda item: one(f"conc{item[0]}-t2-full", item[1], "greedy", 0, "auto",
                                           f"conc{item[0]}-t2"), enumerate(seconds)))
        for i, ids in enumerate(seconds):
            one(f"conc{i}-t2-hit256", ids, "greedy", 256, "auto", f"conc{i}-t2")
        del refs
        status = "CAPTURED"
    except (OSError, RuntimeError, ValueError, TimeoutError) as error:
        status = f"FAIL: {error}"
    finally:
        backend.close()
    report = dict(status=status, engine=str(args.engine), ctx=args.ctx, prompt_tokens=args.prompt_tokens,
                  max_tokens=args.max_tokens, rows=rows, finished=time.time())
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    return 0 if status == "CAPTURED" else 1


STATS = re.compile(r"spec_stats id=(\d+) cached=(\d+) proposed=(\d+) accepted=(\d+) steps=(\d+) .*"
                   r"mtp_origin=(\d+) mtp_continued=(\d)")


def compare_group(name, group):
    """(problems, not compared, no proof) for one group. Rows of one window must be equal; rows of two windows only at
    an equal restore point (cached_tokens), and at least one such pair must exist."""
    problems, skipped, crossed = [], [], 0
    if len({row["prompt_sha256"] for row in group}) != 1:
        return [f"{name}: prompts differ"], skipped, []
    for i, a in enumerate(group):
        for b in group[i + 1:]:
            if a["mtp_prefix"] != b["mtp_prefix"] and a["cached_tokens"] != b["cached_tokens"]:
                skipped.append(f"{name}: {a['label']} (window {a['mtp_prefix']}, restore {a['cached_tokens']}) and "
                               f"{b['label']} (window {b['mtp_prefix']}, restore {b['cached_tokens']})")
                continue
            crossed += a["mtp_prefix"] != b["mtp_prefix"]
            if a["output_ids"] != b["output_ids"]:
                problems.append(f"{name}: outputs differ: {a['label']}={digest(a['output_ids'])[:12]}, "
                                f"{b['label']}={digest(b['output_ids'])[:12]}")
    windows = len({row["mtp_prefix"] for row in group})
    unproven = [f"{name}: no pair of two windows at an equal restore point"] if windows > 1 and not crossed else []
    return problems, skipped, unproven


def check(args):
    report = json.loads(args.output.read_text())
    stats = {}
    for line in args.log.read_text().splitlines():
        if match := STATS.search(line):
            values = [int(value) for value in match.groups()]
            stats[values[0]] = dict(zip(("cached", "proposed", "accepted", "steps", "origin", "continued"), values[1:]))
    problems = [] if report["status"] == "CAPTURED" else [report["status"]]
    rows = {row["label"]: row | dict(stats=stats.get(row["request_id"])) for row in report["rows"]}
    groups = {}
    for row in rows.values():
        groups.setdefault(row["group"], []).append(row)
    not_compared, no_proof = [], []
    for name, group in groups.items():
        found, skipped, unproven = compare_group(name, group)
        problems += found
        not_compared += skipped
        no_proof += unproven
    for row in rows.values():
        if row["stats"] is None:
            problems.append(f"{row['label']}: no spec_stats line")
    # The restored MTP state is the cold state: a frontier hit continues it and drafts exactly as the request that saved it.
    for mode in MODES:
        save, hit, turn = rows.get(f"{mode}-a-save"), rows.get(f"{mode}-a-hit"), rows.get(f"{mode}-b-hit")
        if not (save and hit and turn and save["stats"] and hit["stats"] and turn["stats"]):
            continue
        if not hit["cached_tokens"] or hit["stats"]["continued"] != 1 or hit["stats"]["origin"] != 0:
            problems.append(f"{mode}-a-hit: no frontier hit that continued MTP rows ({hit['stats']}, cached "
                            f"{hit['cached_tokens']})")
        if any(save["stats"][key] != hit["stats"][key] for key in ("proposed", "accepted", "steps")):
            problems.append(f"{mode}-a: drafts after the restore differ from the saving request "
                            f"({save['stats']} against {hit['stats']})")
        if turn["cache_class"] != "replay" or turn["stats"]["continued"] != 1:
            problems.append(f"{mode}-b-hit: the replayed turn did not continue MTP rows ({turn['stats']}, "
                            f"{turn['cache_class']})")
    for row in rows.values():
        if row["label"].endswith("t2-full") and (row["cache_class"] != "replay" or not row["stats"] or
                                                  row["stats"]["continued"] != 1):
            problems.append(f"{row['label']}: the concurrent second turn did not continue MTP rows")
    status = "FAIL" if problems else "NO PROOF" if no_proof else "PASS"
    summary = dict(status=status, problems=problems, no_proof=no_proof, not_compared=not_compared,
                   rows=[dict(label=label, prompt_tokens=row["prompt_tokens"], cached=row["cached_tokens"],
                              cache_class=row["cache_class"], slot_resume=row["slot_resume"], ttft=row["ttft"],
                              decode_seconds=row["decode_seconds"], tokens=len(row["output_ids"]),
                              output_sha256=digest(row["output_ids"])[:16], stats=row["stats"])
                         for label, row in rows.items()])
    print(json.dumps(summary, indent=1))
    return 0 if status == "PASS" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run")
    one.add_argument("engine", type=Path)
    one.add_argument("checkpoint", type=Path)
    one.add_argument("output", type=Path)
    one.add_argument("--prompt-tokens", type=int, default=12000)
    one.add_argument("--max-tokens", type=int, default=96)
    one.add_argument("--ctx", type=int, default=32768)
    one.add_argument("--cache-mib", type=int, default=8192)
    two = commands.add_parser("check")
    two.add_argument("output", type=Path)
    two.add_argument("log", type=Path)
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else check(args)


if __name__ == "__main__":
    raise SystemExit(main())
