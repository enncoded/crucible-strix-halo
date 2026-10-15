#!/usr/bin/env python3
"""Session replay (G15) invariance gate: the same session history gives the same IDs in every serve mode.

With session replay, a later turn's output depends on the request and on the session history (the earlier prompts
and the generated tokens), not on the server mode or load. This gate builds fixed token-level sessions live: each
later prompt is the previous prompt, the generated tokens and a fixed new user chunk. Every later turn must be a
replay hit. One capture is one model load: one mode (serial or batch) and one phase (alone, or with peers that
decode while the target prefills and decodes). `compare` requires all four captures of one pack and equal IDs.

Run each capture under gpu-run with `timeout -k 10 300`. The fresh-request path keeps its own gate
(tools/serve_config_gate.py, --session-replay 0).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.api import STOP_IDS, VOCAB, validate_request
# Rows of one forward (kMaxRows): only a longer prefill chunk runs decode passes inside it.
from server.batch_backend import SMALL_ROW_CHUNK

MODES = ("serial", "batch")
PHASES = ("alone", "peers")
PACKED_OPS = ("step", "verify_step")
# The served default (server/__main__.py --prefill-interleave): batch captures run decode passes between the layers
# of a prompt chunk. The gate passes it explicitly and records it, as the config gate does.
PREFILL_INTERLEAVE = 10
IM_END = 248046
# Fixed sessions. Turn sizes are new user tokens; 120 is below the 256-token MTP warm-up (short suffix path).
SESSIONS = (
    dict(name="greedy-spec", offset=0, turns=(3000, 120, 1500), max_tokens=(48, 48, 48),
         spec=(7, 7, 7), temperature=0.0, seed=1),
    dict(name="sampled-spec", offset=40, turns=(2600, 1500, 120), max_tokens=(48, 48, 48),
         spec=(7, 7, 7), temperature=0.6, seed=11),
    # A spec 0 turn saves a tail without MTP state; the next spec turn restarts the warm-up at the restored length.
    dict(name="policy-change", offset=80, turns=(2200, 120, 900), max_tokens=(40, 40, 40),
         spec=(0, 7, 7), temperature=0.0, seed=3),
    # Small odd limits: a length stop often cuts an accepted window, so the tail is truncated.
    dict(name="window-cut", offset=120, turns=(1800, 200, 120), max_tokens=(13, 17, 21),
         spec=(7, 7, 7), temperature=0.0, seed=5),
)
PEER_TOKENS = 96


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def synthetic_code(offset, lines):
    """Frozen code-like text (the repository changes; this text does not)."""
    return "".join(f"def handler_{offset + i}(value):\n    return value * {(offset + i) % 7} + {(offset + i) % 11}\n"
                   for i in range(lines))


def session_chunks(encode, session):
    """The fixed new-token chunks of one session: chunk 0 is the whole first prompt."""
    code = encode(synthetic_code(session["offset"] * 1000, 4 * sum(session["turns"]) // 10 + 64))
    if len(code) < sum(session["turns"]):
        raise ValueError("not enough synthetic text for the session")
    head = encode("<|im_start|>system\nYou are a coding agent. Answer briefly.<|im_end|>\n<|im_start|>user\n"
                  "Read this code, then name one risk in it.\n")
    tail = encode("<|im_end|>\n<|im_start|>assistant\n")
    user = encode("\n<|im_start|>user\nTool result:\n")
    ask = encode("\nWhat else is risky?")
    chunks, start = [], 0
    for index, size in enumerate(session["turns"]):
        body = code[start:start + size]
        start += size
        chunks.append(head + body + tail if index == 0 else user + body + ask + tail)
    return chunks


def next_prompt(prompt, output, chunk):
    """The client's next prompt: the template closes an answer that stopped at the token limit."""
    closed = output + ([] if output and output[-1] in STOP_IDS else [IM_END])
    return prompt + closed + chunk


def request_for(session, turn, ids, mtp_prefix=None):
    body = dict(prompt=list(ids), max_tokens=session["max_tokens"][turn], spec=session["spec"][turn],
                temperature=session["temperature"], seed=session["seed"] + turn, cache_control="auto")
    if session["temperature"] > 0:
        body |= dict(top_p=0.95)
    if mtp_prefix is not None:
        body |= dict(mtp_prefix=mtp_prefix)
    return validate_request(body, "completion", "crucible")


def collect(backend, ids, request, timeout, times=None, decoding=None):
    """times: a list that gets the time.monotonic() arrival of each token.
    decoding: a threading.Event set at the first token (or at the end), so another request can wait for this decoder."""
    started = time.monotonic()
    job = backend.submit(ids, request)
    out, first = [], None
    try:
        while True:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("request timed out")
            event = job.events.get(timeout=remaining)
            kind = event.get("event")
            if kind == "error":
                raise RuntimeError(f"native serve error: {event.get('error')}")
            if kind == "done":
                if event.get("completion_tokens") != len(out) or event.get("prompt_tokens") != len(ids) or not out:
                    raise RuntimeError("invalid completion counts")
                return dict(output_ids=out, done=event, ttft=first, seconds=time.monotonic() - started)
            token = event.get("id")
            if kind != "token" or type(token) is not int or not 0 <= token < VOCAB or len(out) >= request.max_tokens:
                raise RuntimeError("invalid token event")
            if first is None:
                first = time.monotonic() - started
                if decoding is not None:
                    decoding.set()
            if times is not None:
                times.append(time.monotonic())
            out.append(token)
    except BaseException:
        job.cancelled.set()
        raise
    finally:
        if decoding is not None:
            decoding.set()


def peer_request(encode, index, spec=0):
    """A real chat turn with a long answer. The old 3-id junk prompts stopped after 10, 1 and 1 tokens on the model
    (train 20 chain), so no target chunk ever had 2 live decoders."""
    ids = encode(f"<|im_start|>user\nWrite the numbers from {100 * index + 1} to {100 * index + 400}, one per line, "
                 "with no other text.<|im_end|>\n<|im_start|>assistant\n")
    body = dict(prompt=ids, max_tokens=PEER_TOKENS, spec=spec, temperature=0.0, seed=881 + index, cache_control="bypass")
    return ids, validate_request(body, "completion", "crucible")


def interleave_record(trace, target):
    """The decode passes inside the target's bulk prefill chunks (the engine's `between` hook): their count, the
    most decoders in one pass and the largest such chunk. A pass inside a chunk is a verify_step entry that the batch
    backend adds while the prefill command runs, so its times lie inside the prefill entry's times."""
    chunks = [entry for entry in trace if entry["op"] == "prefill" and entry["request_ids"] == [target]
              and (entry.get("rows") or 0) > SMALL_ROW_CHUNK]
    passes = [(entry, chunk) for chunk in chunks for entry in trace if entry["op"] == "verify_step" and
              target not in entry["request_ids"] and chunk["t_start"] <= entry["t_start"] and entry["t_end"] <= chunk["t_end"]]
    return dict(passes=len(passes), max_decoders=max((len(entry["request_ids"]) for entry, _ in passes), default=0),
                max_chunk_rows=max((chunk["rows"] for _, chunk in passes), default=0))


def max_interleaved_decoders(report):
    return max((row.get("interleave", {}).get("max_decoders", 0) for row in report.get("rows", [])), default=0)


def with_peers(backend, ids, request, peers, timeout, peer_spec=0, encode=None):
    """The peers start first; the target starts when every peer has its first token, so the target's bulk chunk finds
    them as ready decoders (a peer still queued for its own prefill cannot decode inside the chunk)."""
    started = time.monotonic()
    decoding = [threading.Event() for _ in range(peers)]
    def target():
        for event in decoding:
            if not event.wait(timeout=timeout):
                raise TimeoutError("a peer never decoded")
        return collect(backend, ids, request, timeout)
    with ThreadPoolExecutor(max_workers=peers + 1) as pool:
        others = [pool.submit(collect, backend, *peer_request(encode, index, peer_spec), timeout, decoding=event)
                  for index, event in enumerate(decoding)]
        results = [future.result(timeout=2 * timeout + 1) for future in [pool.submit(target), *others]]
    trace = backend.trace.since(started)
    target = results[0]["done"].get("request_id")
    overlap = any(entry["op"] in PACKED_OPS and len(entry["request_ids"]) > 1 and target in entry["request_ids"]
                  for entry in trace)
    return results[0] | dict(peer_count=peers, peer_overlap=overlap, interleave=interleave_record(trace, target),
                             peers=[dict(output_ids=row["output_ids"]) for row in results[1:]])


def run_session(backend, encode, session, phase, peers, timeout, mtp_prefix=None, peer_spec=0):
    chunks = session_chunks(encode, session)
    rows, prompt, output = [], None, None
    for turn, chunk in enumerate(chunks):
        ids = chunk if turn == 0 else next_prompt(prompt, output, chunk)
        request = request_for(session, turn, ids, mtp_prefix)
        row = (with_peers(backend, ids, request, peers, timeout, peer_spec, encode) if phase == "peers"
               else collect(backend, ids, request, timeout))
        expected = 0 if turn == 0 else len(prompt) + len(output) - 1
        rows.append(dict(session=session["name"], turn=turn, prompt_ids_sha256=digest(ids), prompt_tokens=len(ids),
                         expected_cached=expected) | row)
        prompt, output = ids, row["output_ids"]
    return rows


def identity(engine, checkpoint):
    sources = [Path(__file__), *sorted((ROOT / "server").glob("*.py"))]
    metadata = {name: file_digest(Path(checkpoint) / name) for name in ("config.json", "tokenizer.json")
                if (Path(checkpoint) / name).is_file()}
    return dict(engine_sha256=file_digest(engine), checkpoint=str(Path(checkpoint).resolve()),
                checkpoint_metadata=metadata, sessions=digest(SESSIONS),
                source_hashes={str(path.relative_to(ROOT)): file_digest(path) for path in sources})


def checkpoint_encoder(checkpoint):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(checkpoint) / "tokenizer.json"))
    return lambda text: tokenizer.encode(text, add_special_tokens=False).ids


def capture(engine, checkpoint, mode, phase, *, encode, context, prefill, cache_mib, streams, timeout, packed_op,
            prefill_interleave=PREFILL_INTERLEAVE, mtp_prefix=None, peer_spec=0):
    from server.backend import EngineBackend
    from server.batch_backend import BatchedEngineBackend
    settings = dict(context=context, prefill=prefill, cache_mib=cache_mib, streams=streams, packed_op=packed_op,
                    prefill_interleave=prefill_interleave)
    # The MTP window changes drafts only, so the IDs must equal those of the default window (mtp-snapshot).
    if mtp_prefix is not None:
        settings["mtp_prefix"] = mtp_prefix
    # peers with drafts are verify decoders inside the target's chunks. Recorded only when set.
    if peer_spec:
        settings["peer_spec"] = peer_spec
    # The engine switches of this process (the server inherits them): gating_check.py reads CRUCIBLE_PLE_HOST here.
    environment = {name: value for name, value in sorted(os.environ.items()) if name.startswith("CRUCIBLE_")}
    report = dict(status="RUNNING", mode=mode, phase=phase, settings=settings, environment=environment,
                  identity=identity(engine, checkpoint), rows=[])
    backend = None
    try:
        if mode == "serial":
            backend = EngineBackend(engine, checkpoint, context, cache_mib, prefill,
                                    engine_options=("--session-replay", "1"))
        else:
            # interleave_min_rows=0: passes inside chunks of any size, as tools/serve_invariance/capture.py.
            backend = BatchedEngineBackend(engine, checkpoint, context, streams=streams, prefill=prefill,
                                           conversation_cache_mib=cache_mib, packed_op=packed_op, session_replay=True,
                                           prefill_interleave=prefill_interleave, interleave_min_rows=0)
        report["launch"] = list(backend.process.args)
        peers = streams - 1 if phase == "peers" else 0
        for session in SESSIONS:
            report["rows"] += run_session(backend, encode, session, phase, peers, timeout, mtp_prefix, peer_spec)
        if backend.fatal or backend.process.poll() is not None:
            raise RuntimeError("the native serve process stopped before the capture finished")
        report["status"] = "CAPTURED"
    except (OSError, RuntimeError, ValueError, TimeoutError) as error:
        report |= dict(status="FAIL", error=str(error) or type(error).__name__)
    finally:
        if backend is not None:
            backend.close()
    return report


def check_capture(report):
    """Fail-closed checks of one capture; returns a list of problems."""
    problems = []
    if report.get("status") != "CAPTURED":
        return [f"{report.get('mode')}/{report.get('phase')}: capture status {report.get('status')}: {report.get('error')}"]
    launch = report.get("launch", [])
    if launch.count("--session-replay") != 1 or launch[launch.index("--session-replay") + 1] != "1":
        problems.append("the native serve command has no --session-replay 1")
    if report.get("mode") == "batch" and type(report.get("settings", {}).get("prefill_interleave")) is not int:
        problems.append("the batch capture does not record its prefill interleave")
    seen = [(row["session"], row["turn"]) for row in report["rows"]]
    if seen != [(name, turn) for session in SESSIONS for name in [session["name"]] for turn in range(len(session["turns"]))]:
        problems.append("missing or reordered session turns")
    for row in report["rows"]:
        where = f"{report['mode']}/{report['phase']} {row['session']} turn {row['turn']}"
        done = row["done"]
        if row["turn"] == 0:
            if done.get("session_replay"):
                problems.append(f"{where}: a first turn cannot replay")
        elif done.get("session_replay") is not True or done.get("cached_tokens") != row["expected_cached"]:
            problems.append(f"{where}: not a replay hit of the previous turn "
                            f"(cached {done.get('cached_tokens')}, expected {row['expected_cached']})")
        if report["phase"] == "peers":
            if len(row.get("peers", [])) != report["settings"]["streams"] - 1:
                problems.append(f"{where}: peers did not finish")
            if report["mode"] == "batch" and row.get("peer_overlap") is not True:
                problems.append(f"{where}: no packed peer overlap in the native trace")
    if not any(row["done"].get("replay_rows", 0) > 0 for row in report["rows"]):
        problems.append(f"{report['mode']}/{report['phase']}: no replayed tail had pending GDN rows")
    return problems


def compare(reports, min_interleaved_decoders=0):
    """With min_interleaved_decoders K > 0, the batch peers capture must also show a decode pass of at least K
    decoders inside one of the target's bulk prefill chunks. That check does not hide the comparison: the sessions are
    compared in every case. The peer outputs of the serial and the batch peers captures must be equal too (the batch
    peers are the decoders inside the target's chunks)."""
    problems, interleave = [], []
    keys = {(report.get("mode"), report.get("phase")) for report in reports}
    if len(reports) != len(MODES) * len(PHASES) or keys != {(m, p) for m in MODES for p in PHASES}:
        problems.append("compare needs exactly one capture per mode and phase")
    for report in reports:
        problems += check_capture(report)
        if (min_interleaved_decoders and report.get("mode") == "batch" and report.get("phase") == "peers" and
                report.get("status") == "CAPTURED" and max_interleaved_decoders(report) < min_interleaved_decoders):
            interleave.append(f"batch/peers: no interleave pass with {min_interleaved_decoders} or more decoders "
                              f"inside a bulk prefill chunk (most {max_interleaved_decoders(report)})")
    if problems:
        return dict(status="FAIL", problems=problems, sessions=[])
    if len({json.dumps(report["identity"], sort_keys=True) for report in reports}) != 1 or \
            len({json.dumps(report["settings"], sort_keys=True) for report in reports}) != 1:
        return dict(status="FAIL", problems=["captures used different inputs, engines or settings"], sessions=[])
    reference = next(report for report in reports if report["mode"] == "serial" and report["phase"] == "alone")
    peer_reports = [report for report in reports if report["phase"] == "peers"]
    sessions = []
    for index, ref in enumerate(reference["rows"]):
        observations = []
        for report in reports:
            row = report["rows"][index]
            same = row["prompt_ids_sha256"] == ref["prompt_ids_sha256"] and row["output_ids"] == ref["output_ids"]
            first = next((i for i, (a, b) in enumerate(zip(ref["output_ids"], row["output_ids"])) if a != b),
                         None if len(ref["output_ids"]) == len(row["output_ids"]) else
                         min(len(ref["output_ids"]), len(row["output_ids"])))
            observations.append(dict(mode=report["mode"], phase=report["phase"], match=same,
                                     prompt_match=row["prompt_ids_sha256"] == ref["prompt_ids_sha256"],
                                     first_difference=first, replay_rows=row["done"].get("replay_rows"),
                                     cache_class=row["done"].get("cache_class"), ttft=row.get("ttft")))
        peers_match = len({json.dumps(report["rows"][index]["peers"]) for report in peer_reports}) == 1
        sessions.append(dict(session=ref["session"], turn=ref["turn"], tokens=len(ref["output_ids"]),
                             match=all(item["match"] for item in observations) and peers_match,
                             peers_match=peers_match, observations=observations))
    passed = not interleave and all(item["match"] for item in sessions)
    return dict(status="PASS" if passed else "FAIL", problems=interleave, sessions=sessions)


def save(path, report):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".replay-gate-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(report, indent=1) + "\n")
    temporary.replace(path)


def mtp_prefix_rows(text):
    # A range() in `choices` makes argparse print every allowed value (about 2 MB) for one bad value.
    value = int(text)
    if not 0 <= value <= 262144:
        raise argparse.ArgumentTypeError(f"{value} is outside 0..262144")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("capture")
    one.add_argument("engine", type=Path)
    one.add_argument("checkpoint", type=Path)
    one.add_argument("output", type=Path)
    one.add_argument("--mode", choices=MODES, required=True)
    one.add_argument("--phase", choices=PHASES, required=True)
    one.add_argument("--ctx", type=int, default=16384)
    one.add_argument("--prefill", type=int, default=4096)
    one.add_argument("--cache-mib", type=int, default=8192)
    one.add_argument("--streams", type=int, default=4, choices=range(2, 9))
    one.add_argument("--request-timeout", type=float, default=120)
    one.add_argument("--packed-op", choices=PACKED_OPS, default="verify_step")
    one.add_argument("--prefill-interleave", type=int, default=PREFILL_INTERLEAVE, metavar="LAYERS",
                     help="batch mode: decode passes after every LAYERS layers of a prompt chunk (default: the "
                          "served 10; 0 = whole chunks)")
    one.add_argument("--mtp-prefix", type=mtp_prefix_rows, metavar="ROWS",
                     help="the MTP warm-up window of every request (default: the request default)")
    one.add_argument("--peer-spec", type=int, choices=range(0, 8), default=0, metavar="N",
                     help="the draft length of the peer requests (0..7; default 0). N > 0 makes the peers verify "
                          "decoders inside the target's prefill chunks")
    matrix = commands.add_parser("compare")
    matrix.add_argument("captures", type=Path, nargs="+")
    matrix.add_argument("--output", type=Path, required=True)
    matrix.add_argument("--min-interleaved-decoders", type=int, choices=range(0, SMALL_ROW_CHUNK + 1), default=0,
                        metavar="K", help="the batch peers capture needs a pass of K or more decoders inside a "
                        "bulk prefill chunk (0..8; 0, the default, checks nothing)")
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"replay gate: refusing to overwrite {args.output}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "capture":
        report = capture(args.engine.resolve(), args.checkpoint.resolve(), args.mode, args.phase,
                         encode=checkpoint_encoder(args.checkpoint), context=args.ctx,
                         prefill=args.prefill, cache_mib=args.cache_mib, streams=args.streams,
                         timeout=args.request_timeout, packed_op=args.packed_op,
                         prefill_interleave=args.prefill_interleave, mtp_prefix=args.mtp_prefix,
                         peer_spec=args.peer_spec)
        save(args.output, report)
        problems = check_capture(report)
        print(json.dumps(dict(status=report["status"] if not problems else "FAIL", problems=problems)))
        return 0 if not problems else 1
    report = compare([json.loads(path.read_text()) for path in args.captures], args.min_interleaved_decoders)
    save(args.output, report)
    print(json.dumps(dict(status=report["status"], problems=report["problems"])))
    for item in report["sessions"]:
        bad = [f"{o['mode']}/{o['phase']}@{o['first_difference']}" for o in item["observations"] if not o["match"]]
        bad += [] if item["peers_match"] else ["peer outputs"]
        print(f"{item['session']} turn {item['turn']}: {'PASS' if item['match'] else 'FAIL ' + ', '.join(bad)}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
