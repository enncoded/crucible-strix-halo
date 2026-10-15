"""Use the real frontend tokenizer and command workers, without an HTTP listener."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import threading
import time

from bench.suite.runtime_environment import require_known, snapshot
from server.api import STOP_IDS, VOCAB
from server.backend import EngineBackend
from server.batch_backend import BatchedEngineBackend
from .model import CONFIGS, PACKED_OPS, digest, file_digest, peer_extra, prepare_case

ROOT = Path(__file__).resolve().parents[2]


def identity_for(engine, checkpoint, cases):
    inputs = {path: expected for case in cases for path, expected in case.inputs}
    for path, expected in inputs.items():
        if file_digest(path) != expected:
            raise ValueError(f"A frozen capture input changed: {path}")
    environment = snapshot(os.environ)
    require_known(environment)
    sources = [*sorted((ROOT / "server").glob("*.py")), *sorted(Path(__file__).parent.glob("*.py")),
               ROOT / "tools/serve_config_gate.py", ROOT / "bench/suite/runtime_environment.py", ROOT / "tools/gates.py"]
    metadata = {name: file_digest(Path(checkpoint) / name) for name in
                ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
                if (Path(checkpoint) / name).is_file()}
    shards = {path.name: dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
              for path in sorted(Path(checkpoint).glob("*.safetensors"))}
    return dict(engine_sha256=file_digest(engine), checkpoint=str(Path(checkpoint).resolve()),
                checkpoint_metadata=metadata, checkpoint_shards=shards, environment=environment,
                corpus_files={Path(path).name: expected for path, expected in inputs.items()},
                source_hashes={str(path.relative_to(ROOT)): file_digest(path) for path in sources})


def collect(backend, case, timeout):
    started = time.monotonic()
    job = backend.submit(case.ids, case.request, cache_prefix_tokens=case.boundary,
                         shared_prefix_tokens=case.shared_boundary)
    ids = []
    try:
        while True:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError(f"Request timed out: {case.name}")
            event = job.events.get(timeout=remaining)
            kind = event.get("event")
            if kind == "error":
                raise RuntimeError(f"Native serve failed for {case.name}: {event.get('error')}")
            if kind == "done":
                if (type(event.get("completion_tokens")) is not int or event["completion_tokens"] != len(ids) or
                        type(event.get("prompt_tokens")) is not int or event["prompt_tokens"] != len(case.ids) or
                        event.get("finish_reason") not in ("length", "stop") or not ids or
                        (event["finish_reason"] == "length" and len(ids) != case.request.max_tokens) or
                        (event["finish_reason"] == "stop" and ids[-1] not in STOP_IDS)):
                    raise RuntimeError(f"Invalid completion counts: {case.name}")
                return dict(request_sha256=case.request_sha256, prompt_ids_sha256=digest(case.ids),
                            output_ids=ids, done=event, cached_tokens=event.get("cached_tokens"),
                            elapsed_seconds=time.monotonic() - started)
            token = event.get("id")
            if kind != "token" or type(token) is not int or not 0 <= token < VOCAB or len(ids) >= case.request.max_tokens:
                raise RuntimeError(f"Invalid token event: {case.name}")
            ids.append(token)
    except BaseException:
        job.cancelled.set()
        raise


def peer_cases(case, streams, extra=0):
    # Long decode intervals keep peers alive while the target resumes or prefills.
    # These auxiliary requests use frozen IDs and do not use the conversation cache.
    return tuple(prepare_case(f"peer-{index}", dict(messages=[dict(role="user", content="peer")],
                 max_tokens=case.request.max_tokens + extra, spec=0, seed=881 + index, temperature=0,
                 return_token_ids=True, crucible_metrics=True, cache_control="bypass"),
                 _PeerTokenizer(index)) for index in range(streams - 1))


def peer_ids(index, numbers=16):
    """A count ("1, 2, ..., 16, " for peer 0, "101, ..." for peer 1) that the model goes on with at temperature 0.
    The earlier 3-ID peers wrote EOS after 1 to 10 tokens, so on crucible-t13 they ended inside the target's
    interleaved prompt chunks and no packed op held the target with a peer (ci305, Oct 7). These are the Qwen3.8
    tokenizer IDs of the text: digits 15 to 24, "," 11, " " 220."""
    start = 1 + 100 * index
    return [id for number in range(start, start + numbers)
            for id in (*(15 + int(digit) for digit in str(number)), 11, 220)]


class _PeerTokenizer:
    def __init__(self, index): self.index = index
    def encode(self, request): return peer_ids(self.index)
    def cache_prefix_tokens(self, request, ids): return 0
    def shared_prefix_tokens(self, request, ids): return 0


def interleave_extra(case, prefill, every):
    return peer_extra(len(case.ids), prefill, every)


def nested_passes(trace, target_id):
    """Decode passes that ran inside one of the target's prompt chunks."""
    chunks = [entry for entry in trace if entry["op"] == "prefill" and target_id in entry["request_ids"]]
    return sum(1 for entry in trace if entry["op"] == "verify_step" and target_id not in entry["request_ids"] and
               any(chunk["t_start"] <= entry["t_start"] and entry["t_end"] <= chunk["t_end"] for chunk in chunks))


def with_peers(backend, case, streams, timeout, extra=0):
    peers = peer_cases(case, streams, extra)
    barrier = threading.Barrier(streams)
    started = time.monotonic()
    def run(item):
        barrier.wait(timeout=timeout)
        return collect(backend, item, timeout)
    with ThreadPoolExecutor(max_workers=streams) as pool:
        futures = [pool.submit(run, item) for item in (case, *peers)]
        try:
            results = [future.result(timeout=timeout + 1) for future in futures]
        except BaseException:
            backend.close()
            raise
    trace = backend.trace.since(started)
    target_id = results[0]["done"].get("request_id")
    overlap = any(entry["op"] in PACKED_OPS and len(entry["request_ids"]) > 1 and
                  target_id in entry["request_ids"] for entry in trace)
    return results[0] | dict(peer_count=len(peers), streams=streams, peer_overlap=overlap,
                            interleaved_passes=nested_passes(trace, target_id), peers=results[1:], trace=trace)


def capture_cases(engine, checkpoint, cases, config, *, context=65536, prefill=4096, cache_mib=16384,
                  streams=4, timeout=240, selected=None, save=lambda report: None, packed_op="verify_step",
                  session_replay=False, prefill_interleave=10):
    # prefill_interleave: the served default (server/__main__.py), so batch configs run decode passes inside chunks.
    # An engine before G6d (crucible-t6) knows only the packed "step" op; RED captures on it use packed_op="step".
    if packed_op not in PACKED_OPS:
        raise ValueError("Use a known packed batch op.")
    if not 0 <= prefill_interleave <= 47 or (packed_op != "verify_step" and prefill_interleave):
        raise ValueError("Use an interleave of 0 to 47 layers; an engine before G6d needs 0.")
    if config not in CONFIGS or not 4 <= streams <= 8 or not 0 < timeout <= 300 or not 1 <= cache_mib <= 16384:
        raise ValueError("Use a known configuration, four to eight streams, and bounded time and cache settings.")
    if not 1 <= context <= 262144 or not 0 <= prefill <= 4096:
        raise ValueError("Use valid context and prefill bounds.")
    by_name = {case.name: case for case in cases}
    selected = tuple(by_name if selected is None else selected)
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(by_name):
        raise ValueError("Select unique request names from the fixed corpus.")
    identity = identity_for(engine, checkpoint, cases)
    report = dict(status="RUNNING", config=config, selected=list(selected), plan=[case.plan() for case in cases],
                  settings=dict(context=context, prefill=prefill, cache_mib=cache_mib, streams=streams,
                                packed_op=packed_op, session_replay=bool(session_replay),
                                prefill_interleave=prefill_interleave),
                  identity=identity, rows=[], launch=None)
    save(report)
    cache = cache_mib if config.endswith("on") else 0
    # Session replay (the serve default since Oct 4) needs the cache, so cache-off configs never replay.
    replay = bool(session_replay) and bool(cache)
    batch = config.startswith("batch")
    every = prefill_interleave if batch else 0
    backend = None
    try:
        if config.startswith("serial"):
            backend = EngineBackend(engine, checkpoint, context, cache, prefill,
                                    engine_options=("--session-replay", "1") if replay else ())
        else:
            # interleave_min_rows=0: the corpus prompts are short, and the gate checks decode passes inside chunks
            # of any size (the served default skips them below 300 rows; a chunk without passes is the
            # prefill_interleave=0 schedule, which the gate also checks). The setting is not in `settings`: the replay
            # gate needs equal settings across captures, and cached baseline captures do not have the key.
            backend = BatchedEngineBackend(engine, checkpoint, context, streams=streams,
                                            conversation_cache_mib=cache, prefill=prefill, packed_op=packed_op,
                                            session_replay=replay, prefill_interleave=every, interleave_min_rows=0)
        report = report | dict(launch=backend.process.args)
        for name in selected:
            case = by_name[name]
            for warmup in case.warmups:
                row = collect(backend, by_name[warmup], timeout)
                report = _append(report, name, warmup, "warmup", row, save)
            for phase in ("solo", "replay", "peers"):
                row = (with_peers(backend, case, streams, timeout, interleave_extra(case, prefill, every))
                       if phase == "peers" else collect(backend, case, timeout))
                report = _append(report, name, name, phase, row, save)
        if backend.fatal or backend.process.poll() is not None:
            raise RuntimeError("The native serve process stopped before the capture finished.")
        if identity_for(engine, checkpoint, cases) != identity:
            raise ValueError("A frozen capture input changed during execution.")
        report = report | dict(status="CAPTURED")
    except (OSError, RuntimeError, ValueError, queue.Empty) as error:
        report = report | dict(status="FAIL", error=str(error) or "The native request timed out.")
    finally:
        if backend is not None:
            backend.close()
    save(report)
    return report


def _append(report, capture_case, request_case, phase, row, save):
    result = report | dict(rows=[*report["rows"], dict(row, capture_case=capture_case, request_case=request_case, phase=phase)])
    save(result)
    return result
