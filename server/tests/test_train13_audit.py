"""Cross-feature regressions found by the train 13 audit."""
from pathlib import Path
import re
import sys

from server import backend
from server.backend import engine_environment

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "tools"))
import switch_registry  # noqa: E402

# Every CRUCIBLE_* name in engine/ and kernels/, classified for serve (RA-15). "keep": serve passes the name to the
# engine. "strip": serve removes it. The classification is the kServe column of the registry engine/switches.h.
REGISTRY = switch_registry.load()
SWITCHES = {s.name: ("keep" if s.serve else "strip", s.note) for s in REGISTRY}


def engine_names():
    """Every CRUCIBLE_* literal in the engine and kernel sources, also those read through helpers or macros."""
    names = set()
    for directory in ("engine", "kernels"):
        for source in (ROOT / directory).rglob("*"):
            if source.is_file():
                names |= set(re.findall(r"CRUCIBLE_[A-Z0-9_]+", source.read_text(errors="replace")))
    return names


def test_every_engine_switch_is_classified():
    names = engine_names()
    assert sorted(names - set(SWITCHES)) == [], "classify the new switch in SWITCHES"
    assert sorted(set(SWITCHES) - names) == [], "remove the switch that the engine no longer names"
    assert all(kind in ("keep", "strip") and reason for kind, reason in SWITCHES.values())


def test_packed_rows_switch_is_known_to_the_gate_tools():
    # Serve-only, but gate and batch-gate records show it, so a run with 16-row passes cannot pass as default.
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import batch_gate_report
    import gates
    from server import backend
    assert SWITCHES["CRUCIBLE_PACKED_ROWS"][0] == "keep"
    assert "CRUCIBLE_PACKED_ROWS" in backend.ENGINE_KEEP_SWITCHES
    assert "CRUCIBLE_PACKED_ROWS" in gates.RUNTIME_KNOBS
    assert "CRUCIBLE_PACKED_ROWS" in batch_gate_report.RUNTIME_KNOBS


def test_numerics_switches_reach_the_engine():
    # Serve passes them; the engine ignores them when CRUCIBLE_EXACT=1.
    from tools import gates
    numerics = {s.name for s in REGISTRY if s.kind == "numerics"} - {"CRUCIBLE_EXACT"}
    assert all(SWITCHES[name][0] == "keep" for name in gates.NUMERICS_SWITCHES)
    assert sorted(numerics) == sorted(gates.NUMERICS_SWITCHES)


def test_serve_keeps_only_the_classified_keep_switches():
    keep = {name for name, (kind, _) in SWITCHES.items() if kind == "keep"}
    assert getattr(backend, "ENGINE_KEEP_SWITCHES", None) == frozenset(keep)
    unknown = {"CRUCIBLE_NOT_YET_CLASSIFIED": "1", "CRUCIBLE_PI_AI_DIR": "/x"}
    environ = {name: "/tmp/x" for name in SWITCHES} | unknown | {"PATH": "/usr/bin", "HIP_VISIBLE_DEVICES": "0"}
    environment, removed = engine_environment(environ)
    # Strip names and unknown CRUCIBLE_* names leave; keep names and every other variable stay.
    assert removed == sorted(set(environ) - keep - {"PATH", "HIP_VISIBLE_DEVICES"})
    assert set(environment) == keep | {"PATH", "HIP_VISIBLE_DEVICES"}
