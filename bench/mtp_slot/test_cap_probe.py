"""cap_probe.plan: one stage per cap, so the cache never needs more than two frontiers (l7 and l7b were refused)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.mtp_slot import cap_probe  # noqa: E402

END = 195589
WINDOWS = [256, 16384, 32768, 65536, 0]


def frontiers(steps):
    """The frontier (boundary) each step needs or saves, in order."""
    return [(s.label, s.boundary) for s in steps]


def test_builds_grow_and_each_extends_the_last_frontier():
    steps = cap_probe.plan(END, WINDOWS, reps=1)
    builds = [s for s in steps if s.label.startswith("build")]
    cuts = [len_ for len_ in (s.length for s in builds) if len_ != END]
    assert cuts == sorted(cuts) == [END - 65536, END - 32768, END - 16384]
    assert all(s.window == 0 for s in builds)


def test_each_stage_uses_only_its_cap_frontier_and_the_end_frontier():
    steps = cap_probe.plan(END, WINDOWS, reps=1)
    end = END - 300
    stage = None
    for step in steps:
        if step.label.startswith("build") and step.length != END:
            stage = step.boundary
        assert step.boundary in (stage, end), step
        if step.label.startswith("w"):
            assert step.length == END and step.boundary == end


def test_every_window_runs_mirrored_in_each_stage():
    steps = cap_probe.plan(END, WINDOWS, reps=2)
    arms = [s.window for s in steps if s.label.startswith("w")]
    assert arms[:6] == [65536, 256, 0, 0, 256, 65536]
    assert arms.count(16384) == arms.count(32768) == arms.count(65536) == 4
    assert arms.count(0) == arms.count(256) == 12


def test_the_last_arm_of_a_stage_touches_the_cap_frontier():
    # The next build hits that frontier first; the end frontier is then the oldest entry and the one evicted.
    steps = cap_probe.plan(END, WINDOWS, reps=1)
    labels = [s.label for s in steps]
    for index, step in enumerate(steps[1:], 1):
        if step.label.startswith("build") and step.length != END and index > 1:
            assert steps[index - 1].window > 300, labels


def test_no_cap_means_one_stage():
    steps = cap_probe.plan(END, [256, 0], reps=1)
    assert [s.label for s in steps] == ["build-end", "w256-r0", "w0-r0", "w0-r0", "w256-r0"]
