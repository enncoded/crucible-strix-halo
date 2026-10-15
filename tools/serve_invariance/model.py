"""Freeze request semantics and fail closed when the configuration matrix is incomplete."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from server.api import VOCAB, validate_request
from server.batch_scheduler import interleave_passes

CONFIGS = ("serial-cache-off", "serial-cache-on", "batch-cache-off", "batch-cache-on")
# Packed passes that prove live peer overlap: the G6c step and, since G6d, verify_step.
PACKED_OPS = ("step", "verify_step")
PHASES = ("solo", "replay", "peers")
# engine/model/flashnext/geometry.h kMaxRows: a prompt chunk of this many rows or fewer has no decode pass inside it.
INTERLEAVE_MIN_ROWS = 8


def peer_extra(prompt_tokens, prefill, every):
    """Peer tokens that decode passes inside the target's prompt chunks can use up (one per pass, plus the step
    between chunks), so peers still decode beside the target after its prefill."""
    if not prefill or every <= 0:
        return 0
    return -(-prompt_tokens // prefill) * (interleave_passes(every) + 1)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class Case:
    name: str
    body_json: str
    ids: tuple
    boundary: int
    shared_boundary: int
    require_hit: bool = False
    warmups: tuple = ()
    group: str = "synthetic"
    expected_prompt_tokens: int | None = None
    inputs: tuple = ()

    @property
    def request(self):
        return validate_request(json.loads(self.body_json), "chat", "crucible")

    @property
    def request_sha256(self):
        return hashlib.sha256(self.body_json.encode()).hexdigest()

    def plan(self):
        return dict(name=self.name, request_sha256=self.request_sha256, prompt_ids_sha256=digest(self.ids),
                    prompt_tokens=len(self.ids), max_tokens=self.request.max_tokens,
                    require_hit=self.require_hit, warmups=list(self.warmups))


def prepare_case(name, body, tokenizer, *, require_hit=False, warmups=(), group="synthetic", expected_ids=None, inputs=()):
    if not isinstance(name, str) or not name or "seed" not in body:
        raise ValueError("Each fixed request needs a name and an explicit seed.")
    request = validate_request(body, "chat", "crucible")
    ids = tuple(tokenizer.encode(request))
    if not ids or any(type(token) is not int or not 0 <= token < VOCAB for token in ids):
        raise ValueError(f"Invalid prompt IDs: {name}")
    if expected_ids is not None and ids != tuple(expected_ids):
        raise ValueError(f"The frozen bundle prompt IDs changed: {name}")
    boundary = tokenizer.cache_prefix_tokens(request, list(ids))
    shared = tokenizer.shared_prefix_tokens(request, list(ids))
    if any(type(value) is not int or not 0 <= value <= len(ids) for value in (boundary, shared)):
        raise ValueError(f"Invalid cache boundary: {name}")
    return Case(name, canonical(body), ids, boundary, shared, require_hit, tuple(warmups), group,
                len(expected_ids) if expected_ids is not None else None, tuple(inputs))


def load_cases(path, tokenizer, *, verify_prompt_ids=True):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != 1 or len(manifest.get("cases", [])) != 12:
        raise ValueError("The fixed request corpus must contain all twelve cases.")
    inputs = [(str(path), file_digest(path))]
    for name, expected in manifest["files"].items():
        source = path.parent / name
        if source.resolve().parent != path.parent or file_digest(source) != expected:
            raise ValueError(f"A frozen corpus file changed: {name}")
        inputs.append((str(source), expected))
    result = []
    for row in manifest["cases"]:
        body = json.loads((path.parent / row["request_file"]).read_text())
        if digest(body) != row["request_sha256"]:
            raise ValueError(f"A fixed request changed: {row['name']}")
        expected = json.loads((path.parent / row["prompt_ids_file"]).read_text()) if row.get("prompt_ids_file") else None
        case = prepare_case(row["name"], body, tokenizer, require_hit=row["require_hit"], warmups=row["warmups"],
                            group=row["group"], expected_ids=expected if verify_prompt_ids else None, inputs=inputs)
        if not verify_prompt_ids and expected is not None:
            from dataclasses import replace
            case = replace(case, expected_prompt_tokens=len(expected))
        result.append(case)
    names = [case.name for case in result]
    if len(names) != len(set(names)) or sum(name.startswith("bundle-") for name in names) != 4:
        raise ValueError("The fixed corpus has duplicate or missing bundle cases.")
    if any(warmup not in names or warmup == case.name for case in result for warmup in case.warmups):
        raise ValueError("A fixed session has an invalid warmup request.")
    return tuple(result)


def expected_rows(plan, selected):
    by_name = {case["name"]: case for case in plan}
    return {(name, name, phase) for name in selected for phase in PHASES} | {
        (name, warmup, "warmup") for name in selected for warmup in by_name[name]["warmups"]}


def first_difference(left, right):
    return next((index for index, (a, b) in enumerate(zip(left, right)) if a != b),
                min(len(left), len(right)) if len(left) != len(right) else None)


def _validated_rows(reports):
    if not reports:
        raise ValueError("No configuration evidence was supplied.")
    reference = reports[0]
    plan, settings, identity = reference["plan"], reference["settings"], reference["identity"]
    by_name = {case["name"]: case for case in plan}
    if not by_name or len(by_name) != len(plan):
        raise ValueError("The fixed request plan is empty or duplicated.")
    seen, rows = set(), []
    for report in reports:
        config = report["config"]
        if config not in CONFIGS or report.get("status") != "CAPTURED":
            raise ValueError("A configuration capture is invalid or unfinished.")
        if report["plan"] != plan or report["settings"] != settings or report["identity"] != identity:
            raise ValueError("Configuration captures used different frozen inputs or settings.")
        _validate_launch(report)
        selected = report["selected"]
        if not selected or len(selected) != len(set(selected)) or set(selected) - set(by_name):
            raise ValueError("A capture has invalid selected requests.")
        expected = {(config, *key) for key in expected_rows(plan, selected)}
        actual = set()
        for row in report["rows"]:
            key = (config, row["capture_case"], row["request_case"], row["phase"])
            if key in actual or key in seen:
                raise ValueError(f"Duplicate request evidence: {key}")
            actual.add(key)
            request = by_name[row["request_case"]]
            _validate_row(row, request, config, settings)
            rows.append(dict(row, config=config))
        if actual != expected:
            raise ValueError(f"Missing or unexpected request phases: {config}")
        seen |= actual
    expected = {(config, *key) for config in CONFIGS for key in expected_rows(plan, by_name)}
    if seen != expected:
        raise ValueError("The four-configuration request matrix is incomplete.")
    # With the served interleave on, peers must decode inside a target's prompt chunk somewhere. Cache-on peer runs
    # restore the prompt and prefill only a short tail, so the cache-off runs usually carry this evidence.
    if settings.get("prefill_interleave", 0) > 0 and not any(
            row["config"].startswith("batch") and row["phase"] == "peers" and row.get("interleaved_passes", 0) > 0
            for row in rows):
        raise ValueError("No decode pass ran inside a prompt chunk in the batch configurations.")
    return by_name, rows


def _validate_launch(report):
    launch, settings, config = report["launch"], report["settings"], report["config"]
    if not isinstance(launch, list) or len(launch) < 5 or launch[1] != "serve":
        raise ValueError("The capture has no native serve command.")
    for flag, expected in (("--ctx", settings["context"]), ("--prefill", settings["prefill"]),
                           ("--streams", settings["streams"] if config.startswith("batch") else None),
                           ("--conversation-cache-mib", settings["cache_mib"] if config.endswith("on") else None)):
        if expected is None:
            if flag in launch: raise ValueError(f"Unexpected native serve option: {flag}")
        elif launch.count(flag) != 1 or launch[launch.index(flag) + 1] != str(expected):
            raise ValueError(f"The native serve option does not match its configuration: {flag}")
    replay = bool(settings.get("session_replay")) and config.endswith("on")
    if replay != ("--session-replay" in launch) or (replay and (launch.count("--session-replay") != 1 or
                                                                launch[launch.index("--session-replay") + 1] != "1")):
        raise ValueError("The native session replay option does not match its configuration.")


def _validate_row(row, request, config, settings):
    if row["request_sha256"] != request["request_sha256"] or row["prompt_ids_sha256"] != request["prompt_ids_sha256"]:
        raise ValueError("A captured request or its prompt IDs changed.")
    ids, done, cached = row["output_ids"], row["done"], row["cached_tokens"]
    if (not isinstance(ids, list) or not ids or len(ids) > request["max_tokens"] or
            any(type(token) is not int or not 0 <= token < VOCAB for token in ids) or
            type(done.get("completion_tokens")) is not int or done["completion_tokens"] != len(ids) or
            type(done.get("prompt_tokens")) is not int or done["prompt_tokens"] != request["prompt_tokens"] or
            done.get("finish_reason") not in ("length", "stop") or
            (done["finish_reason"] == "length" and len(ids) != request["max_tokens"])):
        raise ValueError("Invalid completion ID or count evidence.")
    if cached != done.get("cached_tokens"):
        raise ValueError("The cache count does not match the native completion.")
    if cached is not None and (type(cached) is not int or not 0 <= cached <= request["prompt_tokens"]):
        raise ValueError("Invalid cache-hit evidence.")
    if config.endswith("off") and cached not in (None, 0):
        raise ValueError("A cache-disabled configuration restored tokens.")
    hit_phase = row["phase"] == "replay" or (request["warmups"] and row["phase"] == "solo")
    if config.endswith("on") and request["require_hit"] and hit_phase and not cached:
        raise ValueError(f"An eligible replay restored no tokens: {request['name']}")
    if config.startswith("batch") and row["phase"] == "peers" and row["peer_overlap"] is not True:
        raise ValueError(f"No live batch peer overlap was proved: {request['name']}")
    if row["phase"] == "peers":
        if row.get("peer_count") != row.get("streams", 0) - 1 or len(row.get("peers", [])) != row["peer_count"]:
            raise ValueError("The peer run did not contain the selected number of peers.")
        if config.startswith("batch") and not any(entry["op"] in PACKED_OPS and len(entry["request_ids"]) > 1 and
                done.get("request_id") in entry["request_ids"] for entry in row.get("trace", [])):
            raise ValueError("The native trace does not prove live target and peer overlap.")
        _validate_peers(row, request, config, settings)


def _validate_peers(row, request, config, settings):
    """Fix 12: the counting peers must stay live for the whole target run, and in batch-cache-off (the
    whole prompt is prefilled) a run must show a decode pass inside a prompt chunk wherever one can run."""
    every = settings.get("prefill_interleave", 0) if config.startswith("batch") else 0
    expected = request["max_tokens"] + peer_extra(request["prompt_tokens"], settings["prefill"], every)
    for peer in row["peers"]:
        done, ids = peer["done"], peer["output_ids"]
        if (done.get("finish_reason") != "length" or not isinstance(ids, list) or len(ids) != expected or
                done.get("completion_tokens") != len(ids)):
            raise ValueError(f"A peer did not end by length at its max_tokens ({expected}): {request['name']}")
    # A pass can run inside a chunk of more than kMaxRows rows (engine InterleaveAllowed). The scheduler prefills the
    # shortest remaining prompt first, so a target shorter than its peers' prompts is prefilled before they decode
    # (short-greedy, 23 tokens, against 55 to 80 for the counting peers): only a longer target must show a pass.
    first_chunk = min(request["prompt_tokens"], settings["prefill"] or request["prompt_tokens"])
    longest_peer = max((peer["done"].get("prompt_tokens", 0) for peer in row["peers"]), default=0)
    if (config == "batch-cache-off" and every > 0 and first_chunk > INTERLEAVE_MIN_ROWS and
            request["prompt_tokens"] > longest_peer and not row.get("interleaved_passes", 0) > 0):
        raise ValueError(f"No decode pass ran inside a prompt chunk of the target: {request['name']}")


def compare(reports, *, required=None, required_prompts=None):
    try:
        plan, rows = _validated_rows(reports)
        if required is not None and {name: case["request_sha256"] for name, case in plan.items()} != required:
            raise ValueError("The captures do not contain the complete frozen request corpus.")
        if required_prompts is not None and any(plan[name]["prompt_ids_sha256"] != expected
                                                for name, expected in required_prompts.items()):
            raise ValueError("A captured bundle prompt differs from its frozen IDs.")
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        return dict(status="FAIL", error=str(error), requests=[])
    requests = []
    for name, case in plan.items():
        observations = [row for row in rows if row["request_case"] == name]
        reference = next(row for row in observations if row["config"] == CONFIGS[0] and
                         row["phase"] == "solo" and row["capture_case"] == name)
        summaries = []
        for row in observations:
            difference = first_difference(reference["output_ids"], row["output_ids"])
            summaries.append(dict(config=row["config"], capture_case=row["capture_case"], phase=row["phase"],
                                  match=difference is None, first_difference=difference,
                                  reference_id=reference["output_ids"][difference] if difference is not None and difference < len(reference["output_ids"]) else None,
                                  actual_id=row["output_ids"][difference] if difference is not None and difference < len(row["output_ids"]) else None,
                                  cached_tokens=row["cached_tokens"], output_ids=row["output_ids"]))
        requests.append(dict(name=name, request_sha256=case["request_sha256"],
                             match=all(row["match"] for row in summaries), observations=summaries))
    return dict(status="PASS" if all(row["match"] for row in requests) else "FAIL", requests=requests)
