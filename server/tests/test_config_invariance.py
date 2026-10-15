"""The gate must reject schedule-dependent IDs and incomplete evidence on CPU."""
import copy
import json
from pathlib import Path
import sys

import pytest

from tools.serve_invariance.model import CONFIGS, load_cases, compare
from tools.serve_invariance.capture import capture_cases


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "bench/suite/serve_invariance/manifest.json"
FAKE = ROOT / "server/tests/fake_config_engine.py"


class FixtureTokenizer:
    """Stable token IDs, with a chat suffix boundary. It is not a model tokenizer."""
    def encode(self, request):
        if len(request.messages) >= 4:
            for name in ("bundle-s0-t0", "bundle-s0-t1", "bundle-s1-t0", "bundle-s2-t0"):
                body = json.loads((CORPUS.parent / f"{name}.request.json").read_text())
                if (request.seed == body["seed"] and len(request.messages) == len(body["messages"]) and
                        request.messages[0]["content"] == body["messages"][0]["content"]):
                    return json.loads((CORPUS.parent / f"{name}.ids.json").read_text())
        content = json.dumps(request.messages, sort_keys=True)
        return [10 + byte for byte in content.encode()][::4] + [30, 31, 32]

    def cache_prefix_tokens(self, request, ids):
        return len(ids) - 3

    def shared_prefix_tokens(self, request, ids):
        return 0


@pytest.fixture(autouse=True)
def no_test_dependency_switches(monkeypatch):
    # CRUCIBLE_PI_AI_DIR and CRUCIBLE_TEST_CKPT only enable optional tests in this suite; the capture policy refuses
    # unknown CRUCIBLE_* names, so clear them here instead of widening the gate allow-list.
    monkeypatch.delenv("CRUCIBLE_PI_AI_DIR", raising=False)
    monkeypatch.delenv("CRUCIBLE_TEST_CKPT", raising=False)


@pytest.fixture
def fake_command(tmp_path):
    launcher = tmp_path / "engine"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    launcher.chmod(0o700)
    return launcher


def small_cases():
    from tools.serve_invariance.model import prepare_case
    body = dict(messages=[dict(role="system", content="abc " * 100), dict(role="user", content="hello")],
                max_tokens=8, spec=0, temperature=0, seed=123, return_token_ids=True, crucible_metrics=True)
    first = prepare_case("first", body, FixtureTokenizer(), require_hit=True)
    second = prepare_case("second", body | dict(messages=body["messages"] + [dict(role="assistant", content="fixed"),
                          dict(role="user", content="next")], temperature=0.7, top_k=32, top_p=0.95),
                          FixtureTokenizer(), require_hit=True, warmups=("first",))
    return (first, second)


def captures(fake_command, monkeypatch, split=False, packed_op="verify_step", **options):
    monkeypatch.setenv("FAKE_CONFIG_SPLIT", "1" if split else "0")
    cases = small_cases()
    return [capture_cases(fake_command, Path("unused"), cases, name, context=2048, prefill=64,
                          cache_mib=16, streams=4, timeout=5, packed_op=packed_op, **options) for name in CONFIGS]


def test_session_replay_default_keeps_fresh_requests_invariant(fake_command, monkeypatch):
    # Replay is the serve default (Oct 4). Requests that repeat no generated tail must still match cache-off.
    reports = captures(fake_command, monkeypatch, session_replay=True)
    for report in reports:
        assert report["settings"]["session_replay"] is True
        assert ("--session-replay" in report["launch"]) == report["config"].endswith("on")
    assert compare(reports)["status"] == "PASS"
    stripped = [dict(report, launch=[item for item in report["launch"] if item != "--session-replay"])
                if report["config"] == "serial-cache-on" else report for report in reports]
    assert compare(stripped)["status"] == "FAIL"
    mixed = [captures(fake_command, monkeypatch)[0], *reports[1:]]
    assert compare(mixed)["status"] == "FAIL"


def packed_ops(reports):
    return {entry["op"] for report in reports if report["config"].startswith("batch")
            for row in report["rows"] for entry in row.get("trace", []) if len(entry.get("request_ids", [])) > 1}


def test_g6d_verify_step_overlap_passes(fake_command, monkeypatch):
    reports = captures(fake_command, monkeypatch)
    assert packed_ops(reports) == {"verify_step"}
    assert compare(reports)["status"] == "PASS"


def test_pre_g6d_engine_red_plan_uses_only_step(fake_command, monkeypatch):
    # crucible-t6 does not know verify_step; its RED captures pack with the G6c step op only.
    reports = captures(fake_command, monkeypatch, split=True, packed_op="step", prefill_interleave=0)
    assert packed_ops(reports) == {"step"}
    failures = {row["config"] for request in compare(reports)["requests"] for row in request["observations"]
                if not row["match"]}
    assert failures == {"serial-cache-on"}


def test_unknown_packed_op_is_rejected(fake_command):
    with pytest.raises(ValueError, match="packed batch op"):
        capture_cases(fake_command, Path("unused"), small_cases(), "batch-cache-off", context=2048, prefill=64,
                      cache_mib=16, streams=4, timeout=5, packed_op="spec_step")


def test_corpus_preserves_all_four_bundle_requests():
    cases = load_cases(CORPUS, FixtureTokenizer(), verify_prompt_ids=False)
    bundle = [case for case in cases if case.name.startswith("bundle-")]
    assert len(bundle) == 4
    assert {case.expected_prompt_tokens for case in bundle} == {12875, 14234}
    assert all(case.request.spec == 7 and case.request.max_tokens == 200 for case in bundle)
    assert any(case.request.temperature > 0 for case in cases)
    assert any(case.warmups for case in cases)
    assert any(case.require_hit for case in cases)


def test_matrix_passes_with_cache_hits_peers_and_sampling(fake_command, monkeypatch):
    reports = captures(fake_command, monkeypatch)
    result = compare(reports)
    assert result["status"] == "PASS"
    assert len(result["requests"]) == 2
    assert all(row["match"] for row in result["requests"])
    assert all(report["status"] == "CAPTURED" for report in reports)
    for report in reports:
        peer_rows = [row for row in report["rows"] if row["phase"] == "peers"]
        if report["config"].startswith("batch"):
            assert all(row["peer_overlap"] for row in peer_rows)
        if report["config"].endswith("on"):
            assert all(row["cached_tokens"] > 0 for row in report["rows"] if row["phase"] == "replay")


def peer_rows(report):
    return [row for row in report["rows"] if row["phase"] == "peers"]


def test_counting_peers_stay_live_beside_the_target(fake_command, monkeypatch):
    # The 3-ID peers wrote EOS after 1 to 10 tokens on crucible-t13. Their decode passes ran inside the target's
    # interleaved prompt chunks, so they ended before the target decoded and ci305 proved no overlap (Oct 7).
    monkeypatch.setenv("FAKE_CONFIG_PEER_STOP", "1")
    reports = captures(fake_command, monkeypatch)
    for report in reports:
        assert all(len(row["peers"]) == row["peer_count"] == 3 for row in peer_rows(report))
        if report["config"].startswith("batch"):
            assert all(row["peer_overlap"] for row in peer_rows(report))
        if report["config"] == "batch-cache-off":
            # Every prompt is prefilled in interleaved chunks, and the peers still decode beside the target after.
            assert all(row["interleaved_passes"] > 0 for row in peer_rows(report))
    assert compare(reports)["status"] == "PASS"


def test_peers_that_end_early_still_fail_closed(fake_command, monkeypatch):
    import tools.serve_invariance.capture as capture
    monkeypatch.setenv("FAKE_CONFIG_PEER_STOP", "1")
    monkeypatch.setattr(capture, "peer_ids", lambda index: [101 + index, 102 + index, 103 + index])
    reports = captures(fake_command, monkeypatch)
    for report in reports:
        assert all(peer["done"]["finish_reason"] == "stop" and len(peer["output_ids"]) == 1
                   for row in peer_rows(report) for peer in row["peers"])
    assert not any(row["peer_overlap"] for report in reports for row in peer_rows(report))
    result = compare(reports)
    # The peers' own evidence fails first (fix 12); the overlap rule alone is the "no-peers" damage case below.
    assert result["status"] == "FAIL" and "did not end by length at its max_tokens" in result["error"]


def damaged(reports, config, change):
    """A deep copy whose first peers row of CONFIG went through CHANGE(row)."""
    reports = copy.deepcopy(reports)
    report = next(report for report in reports if report["config"] == config)
    change(next(row for row in report["rows"] if row["phase"] == "peers"))
    return reports


def test_every_batch_cache_off_peer_run_needs_a_decode_pass_inside_a_prompt_chunk(fake_command, monkeypatch):
    # one run with nested passes must not carry the others. In batch-cache-off every target
    # prefills its whole prompt, so every peers row must show a pass inside a chunk (counting peers stay live).
    reports = captures(fake_command, monkeypatch)
    assert compare(reports)["status"] == "PASS"
    result = compare(damaged(reports, "batch-cache-off", lambda row: row.update(interleaved_passes=0)))
    assert result["status"] == "FAIL" and "inside a prompt chunk" in result["error"]
    # Without the interleave (an engine before G6d) no pass is required.
    whole = captures(fake_command, monkeypatch, prefill_interleave=0)
    assert compare(whole)["status"] == "PASS"


def test_a_target_shorter_than_its_peers_needs_no_nested_pass(fake_command, monkeypatch):
    # The scheduler prefills the shortest prompt first, so a short target (the corpus short cases: 23 tokens, the
    # counting peers 55 to 80) is prefilled before its peers decode and has no pass inside it (ci305 run 2, Oct 7).
    reports = captures(fake_command, monkeypatch)
    def shorter_than_peers(row):
        row["interleaved_passes"] = 0
        for peer in row["peers"]:
            peer["done"]["prompt_tokens"] = row["done"]["prompt_tokens"] + 1
    assert compare(damaged(reports, "batch-cache-off", shorter_than_peers))["status"] == "PASS"


@pytest.mark.parametrize("config", ["serial-cache-off", "batch-cache-on"])
@pytest.mark.parametrize("damage", ["stop", "short", "long"])
def test_a_peer_must_decode_to_its_max_tokens(fake_command, monkeypatch, config, damage):
    # a counting peer ends by length with all its tokens (target max_tokens plus the interleave
    # extra), so it was live for the whole target run; a peer that ended early proves nothing.
    reports = captures(fake_command, monkeypatch)
    def change(row):
        peer = row["peers"][0]
        if damage == "stop": peer["done"]["finish_reason"] = "stop"
        elif damage == "short": peer["output_ids"] = peer["output_ids"][:-1]
        else: peer["output_ids"] = peer["output_ids"] + [7]
        peer["done"]["completion_tokens"] = len(peer["output_ids"])
    result = compare(damaged(reports, config, change))
    assert result["status"] == "FAIL" and "did not end by length at its max_tokens" in result["error"]


def test_peer_prompts_are_counts_in_the_model_vocabulary():
    from tools.serve_invariance.capture import peer_ids
    # IDs of "1, 2, 3," from the Qwen3.8 tokenizer.json (checked on the test machine, Oct 7): digits 15-24, "," 11, " " 220.
    assert peer_ids(0)[:9] == [16, 11, 220, 17, 11, 220, 18, 11, 220]
    assert peer_ids(1)[:6] == [16, 15, 16, 11, 220, 16]
    prompts = [tuple(peer_ids(index)) for index in range(7)]
    assert len(set(prompts)) == 7 and all(prompt[-2:] == (11, 220) for prompt in prompts)


def test_split_fake_fails_only_serial_cache_on(fake_command, monkeypatch):
    result = compare(captures(fake_command, monkeypatch, split=True))
    assert result["status"] == "FAIL"
    failures = [observation for request in result["requests"] for observation in request["observations"]
                if not observation["match"]]
    assert failures
    assert {row["config"] for row in failures} == {"serial-cache-on"}
    assert all(row["first_difference"] == 3 for row in failures)


@pytest.mark.parametrize("damage", ["missing-config", "missing-phase", "duplicate", "not-final", "wrong-hash",
                                      "empty-ids", "bool-id", "no-hit", "no-peers", "engine-mismatch", "settings-mismatch", "wrong-launch", "missing-trace"])
def test_incomplete_or_invalid_evidence_fails_closed(fake_command, monkeypatch, damage):
    reports = copy.deepcopy(captures(fake_command, monkeypatch))
    if damage == "missing-config": reports.pop()
    elif damage == "missing-phase": reports[0]["rows"].pop()
    elif damage == "duplicate": reports.append(reports[0])
    elif damage == "not-final": reports[0]["status"] = "RUNNING"
    elif damage == "wrong-hash": reports[0]["rows"][0]["request_sha256"] = "0" * 64
    elif damage == "empty-ids": reports[0]["rows"][0]["output_ids"] = []
    elif damage == "bool-id": reports[0]["rows"][0]["output_ids"][0] = True
    elif damage == "no-hit":
        for row in reports[1]["rows"]: row["cached_tokens"] = 0
    elif damage == "no-peers":
        for row in reports[2]["rows"]: row["peer_overlap"] = False
    elif damage == "engine-mismatch": reports[0]["identity"]["engine_sha256"] = "0" * 64
    elif damage == "settings-mismatch": reports[0]["settings"]["prefill"] += 1
    elif damage == "wrong-launch": reports[0]["launch"] += ["--streams", "4"]
    elif damage == "missing-trace":
        for row in reports[2]["rows"]:
            if row["phase"] == "peers": row["trace"] = []
    assert compare(reports)["status"] == "FAIL"


def test_truncated_outputs_report_the_first_missing_token(fake_command, monkeypatch):
    reports = copy.deepcopy(captures(fake_command, monkeypatch))
    row = reports[1]["rows"][0]
    row["output_ids"] = row["output_ids"][:-1]
    row["done"]["completion_tokens"] -= 1
    row["done"]["finish_reason"] = "stop"
    result = compare(reports)
    assert result["status"] == "FAIL"
    request = result["requests"][0]
    assert not request["match"]
    mismatch = next(row for row in request["observations"] if row["config"] == "serial-cache-on" and row["phase"] == "solo")
    assert mismatch["first_difference"] == 7


def test_cli_does_not_replace_existing_pending_evidence(tmp_path):
    from tools.serve_config_gate import main
    output = tmp_path / "report.json"
    output.write_text('{"status":"PENDING"}\n')
    assert main(["compare", "absent.json", "--output", str(output)]) == 1
    assert output.read_text() == '{"status":"PENDING"}\n'


def test_comparison_rejects_a_self_consistent_reduced_plan(fake_command, monkeypatch):
    reports = captures(fake_command, monkeypatch)
    required = {"first": small_cases()[0].request_sha256, "second": small_cases()[1].request_sha256,
                "missing": "0" * 64}
    assert compare(reports, required=required)["status"] == "FAIL"


def test_full_frozen_matrix_passes_and_cli_compares_it(fake_command, monkeypatch, tmp_path):
    from tools.serve_config_gate import main
    cases = load_cases(CORPUS, FixtureTokenizer(), verify_prompt_ids=False)
    monkeypatch.setenv("FAKE_CONFIG_SPLIT", "0")
    reports = [capture_cases(fake_command, Path("unused"), cases, config, context=65536, prefill=4096,
                            cache_mib=16, streams=4, timeout=10) for config in CONFIGS]
    paths = []
    for index, report in enumerate(reports):
        path = tmp_path / f"capture-{index}.json"
        path.write_text(json.dumps(report))
        paths.append(str(path))
    output = tmp_path / "matrix.json"
    assert main(["compare", *paths, "--output", str(output)]) == 0
    matrix = json.loads(output.read_text())
    assert matrix["status"] == "PASS"
    assert len(matrix["requests"]) == 12


def test_cli_capture_uses_the_requested_config(fake_command, monkeypatch, tmp_path):
    from tools import serve_config_gate
    monkeypatch.setattr(serve_config_gate, "ModelTokenizer", lambda checkpoint: FixtureTokenizer())
    output = tmp_path / "capture.json"
    result = serve_config_gate.main(["capture", str(fake_command), "unused", str(output),
                                    "--config", "serial-cache-off", "--case", "short-greedy"])
    assert result == 0
    report = json.loads(output.read_text())
    assert report["status"] == "CAPTURED" and report["selected"] == ["short-greedy"]
    assert len(report["plan"]) == 12


def test_cli_rejects_conflicting_selection(tmp_path):
    from tools.serve_config_gate import main
    output = tmp_path / "capture.json"
    assert main(["capture", "absent", "unused", str(output), "--config", "serial-cache-off",
                 "--case", "short-greedy", "--group", "synthetic"]) == 1
    assert json.loads(output.read_text())["status"] == "FAIL"


def test_engine_change_during_capture_is_not_qualified(fake_command, monkeypatch):
    monkeypatch.setenv("FAKE_CONFIG_SPLIT", "0")
    def change(report):
        if report["launch"] is None:
            fake_command.write_text(fake_command.read_text() + "# changed\n")
    report = capture_cases(fake_command, Path("unused"), small_cases(), "serial-cache-off", context=2048,
                           prefill=64, cache_mib=16, streams=4, timeout=5, save=change)
    assert report["status"] == "FAIL"
    assert "changed" in report["error"]


@pytest.mark.parametrize("events", [[dict(event="error", error="bad")], [dict(event="token", id=True)],
                                     [dict(event="other")], [dict(event="done", completion_tokens=0)], []])
def test_invalid_native_output_never_becomes_evidence(events):
    from tools.serve_invariance.capture import collect
    from server.tests.serve_fixtures import ScriptBackend
    with pytest.raises(Exception):
        collect(ScriptBackend(events), small_cases()[0], timeout=0.01)


def test_repeated_shards_are_rejected(fake_command, monkeypatch):
    reports = captures(fake_command, monkeypatch)
    assert compare([*reports, reports[0]])["status"] == "FAIL"
    shards = []
    for report in reports:
        for name in report["selected"]:
            shards.append(report | dict(selected=[name], rows=[row for row in report["rows"] if row["capture_case"] == name]))
    assert compare(shards)["status"] == "PASS"


def test_corpus_file_hash_changes_are_rejected(tmp_path):
    manifest = json.loads(CORPUS.read_text())
    manifest["cases"][0]["request_sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_cases(path, FixtureTokenizer(), verify_prompt_ids=False)



def interleaved_passes(reports):
    return sum(row.get("interleaved_passes", 0) for report in reports if report["config"].startswith("batch")
               for row in report["rows"])


def test_batch_configs_run_the_served_prefill_interleave(fake_command, monkeypatch):
    # The served default (--prefill-interleave 10) must be what ci305 checks: peers decode inside prompt chunks
    # and still pack with the target after its prefill.
    reports = captures(fake_command, monkeypatch)
    assert {report["settings"]["prefill_interleave"] for report in reports} == {10}
    assert interleaved_passes(reports) > 0
    assert compare(reports)["status"] == "PASS"
    whole = captures(fake_command, monkeypatch, prefill_interleave=0)
    assert interleaved_passes(whole) == 0 and compare(whole)["status"] == "PASS"
    # Evidence without a nested pass fails while the interleave is on.
    stripped = [dict(report, rows=[dict(row, interleaved_passes=0) for row in report["rows"]]) for report in reports]
    assert "inside a prompt chunk" in compare(stripped)["error"]
