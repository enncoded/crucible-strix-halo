"""probe.py check, rule of Oct 10: rows of two windows must give equal IDs only at an equal restore point. Rows of
one window keep the old rule (every row of the group equals the others)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.mtp_snapshot.probe import compare_group  # noqa: E402


def row(label, window, cached, ids, prompt="p"):
    return dict(label=label, mtp_prefix=window, cached_tokens=cached, output_ids=ids, prompt_sha256=prompt)


def test_two_windows_at_one_restore_point_must_match():
    problems, skipped, unproven = compare_group("b", [row("hit", 0, 12133, [1, 2]), row("hit256", 256, 12133, [1, 3])])
    assert problems and "outputs differ" in problems[0] and skipped == [] and unproven == []


def test_two_windows_at_different_restore_points_are_no_proof():
    # Review C1: a group of two windows with no pair at an equal restore point proves nothing, so it is NO PROOF.
    problems, skipped, unproven = compare_group("b", [row("hit", 0, 12141, [1, 2]), row("hit256", 256, 12133, [1, 3])])
    assert problems == [] and skipped == ["b: hit (window 0, restore 12141) and hit256 (window 256, restore 12133)"]
    assert unproven == ["b: no pair of two windows at an equal restore point"]


def test_one_window_group_needs_no_cross_window_pair():
    assert compare_group("a", [row("cold", 0, 0, [1]), row("hit", 0, 9, [1])]) == ([], [], [])


def test_check_gives_no_proof_and_exit_1(tmp_path):
    import json
    from types import SimpleNamespace
    from bench.mtp_snapshot.probe import check
    rows = [dict(row("b-hit", 0, 12141, [1, 2]), group="b", request_id=1, cache_class="replay", slot_resume=False,
                 prompt_tokens=9, ttft=0.1, decode_seconds=1.0),
            dict(row("b-hit256", 256, 12133, [1, 3]), group="b", request_id=2, cache_class="replay",
                 slot_resume=False, prompt_tokens=9, ttft=0.1, decode_seconds=1.0)]
    out, log = tmp_path / "o.json", tmp_path / "e.log"
    out.write_text(json.dumps(dict(status="CAPTURED", rows=rows)))
    log.write_text("".join(f"spec_stats id={i} cached=0 proposed=1 accepted=1 steps=1 x mtp_origin=0 "
                           f"mtp_continued=1\n" for i in (1, 2)))
    assert check(SimpleNamespace(output=out, log=log)) == 1


def test_one_window_keeps_the_old_rule_at_any_restore_point():
    problems, _, _ = compare_group("a", [row("cold", 0, 0, [1, 2]), row("hit", 0, 12000, [1, 3])])
    assert problems and "outputs differ" in problems[0]
    problems, skipped, unproven = compare_group("a", [row("ref", 256, 0, [1, 2]), row("cold", 0, 0, [1, 2]),
                                                      row("hit", 0, 12000, [1, 2])])
    assert problems == [] and len(skipped) == 1 and unproven == []


def test_prompts_must_be_equal():
    problems, _, _ = compare_group("c", [row("cold", 0, 0, [1], "p"), row("ref", 256, 0, [1], "q")])
    assert problems == ["c: prompts differ"]
