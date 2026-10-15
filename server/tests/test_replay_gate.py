"""CPU proof of the session replay gate (tools/serve_replay_gate.py) with a fake native engine."""
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from tools import serve_replay_gate as gate

FAKE = Path(__file__).parent / "fake_replay_engine.py"


def encode(text):
    return [500 + ord(character) % 997 for character in text]


@pytest.fixture
def fake_engine(tmp_path, monkeypatch):
    popen = subprocess.Popen
    def start(arguments, **kwargs):
        return popen([sys.executable, str(FAKE), *[str(item) for item in arguments[1:]]], **kwargs)
    monkeypatch.setattr("server.backend.subprocess.Popen", start)
    monkeypatch.setattr(gate, "SESSIONS", tuple(dict(session, turns=tuple(size // 40 for size in session["turns"]))
                                               for session in gate.SESSIONS))
    monkeypatch.setattr(gate, "PEER_TOKENS", 24)
    engine = tmp_path / "engine"
    engine.write_bytes(b"fake engine")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    return engine, checkpoint


def captures(engine, checkpoint, **options):
    return [gate.capture(engine, checkpoint, mode, phase, encode=encode, context=8192, prefill=64, cache_mib=64,
                         streams=4, timeout=30, packed_op="verify_step", **options)
            for mode in gate.MODES for phase in gate.PHASES]


def test_same_history_same_ids_passes(fake_engine):
    reports = captures(*fake_engine)
    for report in reports:
        assert gate.check_capture(report) == [], report.get("error")
    result = gate.compare(reports)
    assert result["status"] == "PASS"
    assert len(result["sessions"]) == sum(len(session["turns"]) for session in gate.SESSIONS)


def test_batch_replay_difference_fails(fake_engine, monkeypatch):
    monkeypatch.setenv("FAKE_REPLAY_FAULT", "batch")
    result = gate.compare(captures(*fake_engine))
    assert result["status"] == "FAIL"
    bad = [item for item in result["sessions"] if not item["match"]]
    assert bad and all(item["turn"] > 0 for item in bad)
    assert {o["mode"] for item in bad for o in item["observations"] if not o["match"]} == {"batch"}


def test_a_turn_without_a_replay_hit_fails_closed(fake_engine, monkeypatch):
    monkeypatch.setenv("FAKE_REPLAY_FAULT", "nohit")
    reports = captures(*fake_engine)
    assert any("not a replay hit" in problem for problem in gate.check_capture(reports[0]))
    assert gate.compare(reports)["status"] == "FAIL"


def test_incomplete_matrix_and_mixed_settings_fail(fake_engine):
    reports = captures(*fake_engine)
    assert gate.compare(reports[:3])["status"] == "FAIL"
    changed = [dict(reports[0], settings=dict(reports[0]["settings"], cache_mib=32)), *reports[1:]]
    assert gate.compare(changed)["status"] == "FAIL"


def test_next_prompt_closes_a_length_stop():
    assert gate.next_prompt([1, 2], [5, 6], [9]) == [1, 2, 5, 6, gate.IM_END, 9]
    stop = next(iter(gate.STOP_IDS))
    assert gate.next_prompt([1], [5, stop], [9]) == [1, 5, stop, 9]


def test_batch_captures_record_the_served_prefill_interleave(fake_engine):
    # The server's default is --prefill-interleave 10 (decode-interleave 8532b03). The gate passes it to the batch
    # backend explicitly and records the value it tests; a batch capture without the value fails closed.
    reports = captures(*fake_engine)
    assert gate.PREFILL_INTERLEAVE == 10
    assert {report["settings"]["prefill_interleave"] for report in reports} == {10}
    assert all(gate.check_capture(report) == [] for report in reports)
    batch = next(report for report in reports if report["mode"] == "batch")
    unrecorded = dict(batch, settings={k: v for k, v in batch["settings"].items() if k != "prefill_interleave"})
    assert any("prefill interleave" in problem for problem in gate.check_capture(unrecorded))
    whole = captures(*fake_engine, prefill_interleave=0)
    assert {report["settings"]["prefill_interleave"] for report in whole} == {0}
    assert gate.compare(whole)["status"] == "PASS"


def test_mtp_prefix_reaches_every_request_and_is_recorded(fake_engine):
    # mtp-snapshot: the full MTP window (0) must give the IDs of the default window. The default capture sends no
    # mtp_prefix field and records none, so its request bytes stay those of the older gate.
    sent = []
    submit = gate.collect
    def spy(backend, ids, request, timeout, **options):
        if request.cache_control == "auto":  # session turns; the peers are plain bypass requests
            sent.append(request.mtp_prefix)
        return submit(backend, ids, request, timeout, **options)
    with patch.object(gate, "collect", spy):
        full = captures(*fake_engine, mtp_prefix=0)
    assert sent and set(sent) == {0}
    assert {report["settings"]["mtp_prefix"] for report in full} == {0}
    assert gate.compare(full)["status"] == "PASS"
    default = captures(*fake_engine)
    assert all("mtp_prefix" not in report["settings"] for report in default)
    assert [row["output_ids"] for row in full[0]["rows"]] == [row["output_ids"] for row in default[0]["rows"]]
    assert gate.request_for(gate.SESSIONS[0], 0, [1, 2]).mtp_prefix == 256


def test_mtp_prefix_range_error_is_short(capsys):
    # argparse printed every allowed value (about 2 MB) for a bad choices=range(...) value.
    with pytest.raises(SystemExit):
        gate.main(["capture", "e", "c", "o", "--mode", "serial", "--phase", "alone", "--mtp-prefix", "999999999"])
    err = capsys.readouterr().err
    assert len(err) < 2000 and "mtp-prefix" in err


def test_collect_records_one_arrival_time_per_token():
    import queue
    import threading
    from types import SimpleNamespace
    events = queue.Queue()
    for item in (dict(event="token", id=7), dict(event="token", id=8),
                 dict(event="done", completion_tokens=2, prompt_tokens=3)):
        events.put(item)
    job = SimpleNamespace(events=events, cancelled=threading.Event())
    backend = SimpleNamespace(submit=lambda ids, request: job)
    times = []
    result = gate.collect(backend, [1, 2, 3], SimpleNamespace(max_tokens=4), 5, times=times)
    assert result["output_ids"] == [7, 8]
    assert len(times) == 2 and times[0] <= times[1]
    assert result["ttft"] <= result["seconds"]


def test_peer_spec_reaches_every_peer_and_is_recorded(fake_engine):
    # the peers decode as verify decoders (spec > 0) while the target prefills. The
    # default capture keeps spec 0 peers and records no peer_spec, so its settings stay those of the older gate.
    sent = []
    submit = gate.collect
    def spy(backend, ids, request, timeout, **options):
        if request.cache_control == "bypass":
            sent.append(request.spec)
        return submit(backend, ids, request, timeout, **options)
    with patch.object(gate, "collect", spy):
        spec = gate.capture(*fake_engine, "batch", "peers", encode=encode, context=8192, prefill=64, cache_mib=64,
                            streams=4, timeout=30, packed_op="verify_step", peer_spec=7)
    assert sent and set(sent) == {7} and spec["settings"]["peer_spec"] == 7
    assert gate.check_capture(spec) == []
    assert all("peer_spec" not in report["settings"] for report in captures(*fake_engine))


def test_batch_peers_record_the_decoders_of_each_interleaved_bulk_chunk(fake_engine):
    # The `between` hook: a target chunk above kMaxRows rows runs verify passes of the live peers inside it.
    report = gate.capture(*fake_engine, "batch", "peers", encode=encode, context=8192, prefill=64, cache_mib=64,
                          streams=4, timeout=30, packed_op="verify_step")
    seen = [row["interleave"] for row in report["rows"]]
    assert all(set(item) == {"passes", "max_decoders", "max_chunk_rows"} for item in seen)
    assert any(item["passes"] >= 1 and item["max_decoders"] >= 2 and item["max_chunk_rows"] > 8 for item in seen)
    assert gate.max_interleaved_decoders(report) >= 2


def test_compare_can_require_interleaved_verify_decoders(fake_engine):
    reports = captures(*fake_engine, peer_spec=7)
    assert gate.compare(reports, min_interleaved_decoders=2)["status"] == "PASS"
    assert gate.compare(reports, min_interleaved_decoders=9)["status"] == "FAIL"
    none = [dict(r, rows=[dict(row, interleave=dict(passes=0, max_decoders=0, max_chunk_rows=0)) for row in r["rows"]])
            for r in reports]
    result = gate.compare(none, min_interleaved_decoders=2)
    assert result["status"] == "FAIL" and any("interleave" in problem for problem in result["problems"])
    assert gate.compare(none)["status"] == "PASS"  # without the option the old rule stands


def test_capture_records_its_engine_switches(fake_engine, monkeypatch):
    # The full CPU run sets CRUCIBLE_TEST_CKPT and CRUCIBLE_PI_AI_DIR (test inputs, not switches); start clean.
    import os
    for name in [name for name in os.environ if name.startswith("CRUCIBLE_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("CRUCIBLE_PLE_HOST", "1")
    monkeypatch.setenv("NOT_A_SWITCH", "x")
    report = gate.capture(*fake_engine, "serial", "alone", encode=encode, context=8192, prefill=64, cache_mib=64,
                          streams=4, timeout=30, packed_op="verify_step")
    assert report["environment"] == {"CRUCIBLE_PLE_HOST": "1"}


def test_peer_spec_and_interleave_options_parse():
    with pytest.raises(SystemExit):
        gate.main(["capture", "e", "c", "o", "--mode", "batch", "--phase", "peers", "--peer-spec", "8"])
    with pytest.raises(SystemExit):
        gate.main(["compare", "a", "--output", "o", "--min-interleaved-decoders", "-1"])


def test_the_bulk_chunk_bound_is_the_backend_value():
    # one definition of the small-row chunk (kMaxRows), the batch backend's.
    from server import batch_backend
    assert gate.SMALL_ROW_CHUNK is batch_backend.SMALL_ROW_CHUNK


def test_peers_are_chat_prompts_that_decode_to_their_token_limit(fake_engine):
    # Train 20 chain (replay305-plehost FAIL): 3-id junk peers stopped after 10, 1 and 1 tokens, so every bulk chunk
    # had at most 1 live decoder. The peers are real chat turns now; the fake stops a junk prompt at once, as the model.
    ids, request = gate.peer_request(encode, 1, 7)
    assert len(ids) >= 8 and ids[-len(encode("<|im_start|>assistant\n")):] == encode("<|im_start|>assistant\n")
    assert request.spec == 7 and request.max_tokens == gate.PEER_TOKENS and request.cache_control == "bypass"
    assert len({tuple(gate.peer_request(encode, index)[0]) for index in range(8)}) == 8
    report = gate.capture(*fake_engine, "batch", "peers", encode=encode, context=8192, prefill=64, cache_mib=64,
                          streams=4, timeout=30, packed_op="verify_step", peer_spec=7)
    assert gate.check_capture(report) == []
    assert all(len(peer["output_ids"]) == gate.PEER_TOKENS for row in report["rows"] for peer in row["peers"])


def test_the_target_starts_after_every_peer_decodes(fake_engine):
    # The target's bulk chunk must find the peers ready (decoding), not still queued for their own prefill.
    starts, firsts = [], []
    submit = gate.collect
    def spy(backend, ids, request, timeout, **options):
        started = time.monotonic()
        if request.cache_control == "auto":
            starts.append(started)
        result = submit(backend, ids, request, timeout, **options)
        if request.cache_control == "bypass":
            firsts.append(started + result["ttft"])
        return result
    with patch.object(gate, "collect", spy):
        for mode in gate.MODES:
            gate.capture(*fake_engine, mode, "peers", encode=encode, context=8192, prefill=64, cache_mib=64,
                         streams=4, timeout=30, packed_op="verify_step", peer_spec=7)
    turns = sum(len(session["turns"]) for session in gate.SESSIONS)
    assert len(starts) == 2 * turns and len(firsts) == 3 * len(starts)
    assert all(start >= max(firsts[3 * i:3 * i + 3]) for i, start in enumerate(starts))


def test_compare_proves_the_peer_outputs_equal_and_keeps_the_sessions_on_an_interleave_fail(fake_engine):
    reports = captures(*fake_engine, peer_spec=7)
    result = gate.compare(reports, min_interleaved_decoders=2)
    assert result["status"] == "PASS" and all(item["peers_match"] for item in result["sessions"])
    batch = next(index for index, r in enumerate(reports) if r["mode"] == "batch" and r["phase"] == "peers")
    rows = [dict(row) for row in reports[batch]["rows"]]
    peers = rows[1]["peers"]
    rows[1] = dict(rows[1], peers=[dict(output_ids=[*peers[0]["output_ids"][:-1], 7]), *peers[1:]])
    changed = [*reports[:batch], dict(reports[batch], rows=rows), *reports[batch + 1:]]
    result = gate.compare(changed)
    assert result["status"] == "FAIL" and [item["peers_match"] for item in result["sessions"]].count(False) == 1
    none = [dict(r, rows=[dict(row, interleave=dict(passes=0, max_decoders=0, max_chunk_rows=0)) for row in r["rows"]])
            for r in reports]
    result = gate.compare(none, min_interleaved_decoders=2)
    assert result["status"] == "FAIL" and result["sessions"] and all(item["match"] for item in result["sessions"])
