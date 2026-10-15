"""Ratchet R1: layer loops in the engine.

Three counts, per tracked engine/ file, that may only go down:
- trunk: `for` or `while` loops with a `kLayers` (or `kMtpLayer`) bound, either side of the comparison, whose
  body runs a layer body (`Block(` or `PrefillBlock(`). In LOOP_HOME every bounded loop is a trunk loop, because
  `RunTrunk` runs the pass's lambdas.
- mtp: one-layer drivers, calls `Block(kLayers, ...)` or `PrefillBlock(kLayers, ...)` (or `kMtpLayer`);
- walker: other bounded loops outside engine/model/ (the model owns its tensor walk, so walkers there do not
  count).
A new site fails test_no_new_layer_loop; a count below ALLOWED fails the stale test, so each cleanup lowers
ALLOWED. Target: one trunk loop, `RunTrunk` in LOOP_HOME, with no MTP driver there; 0 elsewhere.
Run `python3 tests/test_layer_loops.py` to print the current counts.
A correct RunTrunk counts as a trunk loop in LOOP_HOME, so the change that adds it also adds
`LOOP_HOME: {'trunk': 1}` to ALLOWED and removes the old trunk entries: the ratchet then shows 4 -> 1.

Limits (written, not tested): the scan reads names, not meaning. A local alias of the bound
(`const int n = kLayers;`), a `#define` alias, a range-for over `m_.L`, a `do ... while` loop, a body behind a
new function name, and loops in bench/ are not seen. `#if 0` blocks count. Reviewers check these by hand.
"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_model_geometry import SUFFIXES, blank_lexemes, source_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = "engine/model/"
LOOP_HOME = "engine/runtime/layer_loop.h"
LOOP = re.compile(r"\b(?:for|while)\s*\(")
BOUND = re.compile(r"<=?\s*(?:kLayers|kMtpLayer)\b|\b(?:kLayers|kMtpLayer)\s*>=?(?!>)")
# `sha256_detail::Block(` and `vision.Block(` are not layer bodies.
BODY_CALL = re.compile(r"(?<![\w:.>])(?:Block|PrefillBlock)\s*\(")
MTP_CALL = re.compile(r"(?<![\w:.>])(?:Block|PrefillBlock)\s*\(\s*(?:kLayers|kMtpLayer)\b")
ADVICE = {
    "trunk": "no new layer loop: run the pass through the existing forward loops (RunTrunk)",
    "mtp": "no new one-layer MTP driver: extend Mtp, MtpSpans or MtpWarm",
    "walker": "no new loop over the layers outside engine/model/: walk them in the model code (engine/model/)",
}


def _match(text, start, open_ch, close_ch):
    """Index after the bracket that closes the one at text[start]."""
    depth = 0
    for i in range(start, len(text)):
        depth += (text[i] == open_ch) - (text[i] == close_ch)
        if depth == 0:
            return i + 1
    return len(text)


def for_loops(code):
    """Yield (header, body) of each `for` or `while` statement in code without comments or strings."""
    for m in LOOP.finditer(code):
        head_end = _match(code, m.end() - 1, "(", ")")
        i = head_end
        while i < len(code) and code[i].isspace():
            i += 1
        end = _match(code, i, "{", "}") if code[i:i + 1] == "{" else code.find(";", i) + 1
        yield code[m.end():head_end - 1], code[i:end]


def count_file(rel, text):
    """{kind: count} for one file."""
    _, bare = blank_lexemes(text)
    counts = {"trunk": 0, "mtp": len(MTP_CALL.findall(bare)), "walker": 0}
    for header, body in for_loops(bare):
        if not BOUND.search(header):
            continue
        if rel == LOOP_HOME or BODY_CALL.search(body):
            counts["trunk"] += 1
        elif not rel.startswith(MODEL_DIR):
            counts["walker"] += 1
    return {k: v for k, v in counts.items() if v}


def scan(root=ROOT):
    found = {}
    for rel, path in source_files(root, ("engine",), SUFFIXES):
        counts = count_file(rel, path.read_text())
        if counts:
            found[rel] = counts
    return found


def loop_home_errors(root=ROOT):
    """After an earlier change: LOOP_HOME holds RunTrunk with exactly one trunk loop and no MTP driver."""
    home = root / LOOP_HOME
    if not home.exists():
        return []
    _, bare = blank_lexemes(home.read_text())
    if not re.search(r"\bRunTrunk\s*\(", bare):
        return [f"{LOOP_HOME}: no RunTrunk"]
    counts = count_file(LOOP_HOME, home.read_text())
    errors = []
    if counts.get("trunk", 0) != 1:
        n = counts.get("trunk", 0)
        errors.append(f"{LOOP_HOME}: {n} trunk loop{'' if n == 1 else 's'}, want 1")
    if counts.get("mtp", 0):
        n = counts["mtp"]
        errors.append(f"{LOOP_HOME}: {n} MTP driver{'' if n == 1 else 's'}, want 0")
    return errors


# Frozen on train 19: trunk 4, mtp 3, walker 6.
# the 4 trunk loops (Forward, ForwardBatch, ForwardVerifySpans, Prefill) call RunTrunk in LOOP_HOME.
ALLOWED = {
    'engine/engine.hip': {'mtp': 1, 'walker': 2},
    'engine/runtime/batch_spec_forward.h': {'mtp': 1},
    'engine/runtime/layer_loop.h': {'trunk': 1},
    'engine/runtime/prefill_engine.inc': {'mtp': 1, 'walker': 3},
    'engine/runtime/sequence_state.h': {'walker': 1},
}


def totals(found):
    out = {"trunk": 0, "mtp": 0, "walker": 0}
    for counts in found.values():
        for kind, n in counts.items():
            out[kind] += n
    return out


def _write(root, files):
    for rel, text in files.items():
        Path(root, rel).parent.mkdir(parents=True, exist_ok=True)
        Path(root, rel).write_text(text)


class LayerLoopRatchet(unittest.TestCase):
    def test_no_new_layer_loop(self):
        new = [f"{path} {kind}: {n} > {ALLOWED.get(path, {}).get(kind, 0)} ({ADVICE[kind]})"
               for path, counts in scan().items() for kind, n in counts.items()
               if n > ALLOWED.get(path, {}).get(kind, 0)]
        self.assertEqual(new, [], "structure rules, an earlier change")

    def test_allow_list_has_no_stale_entry(self):
        found = scan()
        stale = [f"{path} {kind}: allowed {n}, found {found.get(path, {}).get(kind, 0)}"
                 for path, counts in ALLOWED.items() for kind, n in counts.items()
                 if found.get(path, {}).get(kind, 0) < n]
        self.assertEqual(stale, [], "fixed sites: lower or remove these ALLOWED entries")

    def test_trunk_loop_exists(self):
        # Until S1, the trunk loops are the ALLOWED ones; after it, RunTrunk in LOOP_HOME holds the only one.
        self.assertEqual(loop_home_errors(), [])
        if not (ROOT / LOOP_HOME).exists():
            self.assertGreater(totals(scan())["trunk"], 0, f"no trunk loop and no {LOOP_HOME}")

    def test_scan_kinds(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, {
                "engine/a.hip": (
                    "for (int l = 0; l < kLayers; ++l) {\n  if (PleBefore(l)) Ple();\n  Block(l, m.L[l]);\n}\n"
                    "for (int l=0; l<kLayers; ++l) PrefillBlock(l, x);\n"
                    "for (int l = 0; l <= kLayers; ++l) alloc(l);\n"
                    "for (int i = 0; i < n; ++i) Block(i);\n"
                    "Block(kLayers, m.mtp.L); PrefillBlock(kMtpLayer, p);\n"
                    "// for (int l = 0; l < kLayers; ++l) Block(l);\n"
                    'f("for (int l = 0; l < kLayers; ++l) Block(l);");\n'
                    "for (int l = 0; l < kLayers; ++l) sha256_detail::Block(h, p);\n"),
                "engine/model/m/model.h": "for (int l = 0; l < kLayers; ++l) Load(l);\n",
                LOOP_HOME: "for (int l = 0; l < kLayers; ++l) { pre(l); Block(l); }\n",
            })
            self.assertEqual(scan(Path(tmp)), {"engine/a.hip": {"trunk": 2, "mtp": 2, "walker": 2},
                                               LOOP_HOME: {"trunk": 1}})


    def test_scan_loop_forms(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, {
                "engine/a.cu": "for (int l = 0; kLayers > l; ++l) Block(l, x);\n"
                               "int l = 0; while (l < kLayers) { PrefillBlock(l, p); ++l; }\n"
                               "for (int l = 0; l <= kMtpLayer; ++l) alloc(l);\n",
                "engine/b.hpp": "for (int l = 0; l < kLayers; ++l) Block(l, x);\n",
            })
            self.assertEqual(scan(Path(tmp)), {"engine/a.cu": {"trunk": 2, "walker": 1},
                                               "engine/b.hpp": {"trunk": 1}})

    def test_loop_home_includes_only_the_model_config(self):
        home = ROOT / LOOP_HOME
        self.assertTrue(home.exists(), f"{LOOP_HOME} holds RunTrunk")
        self.assertEqual(re.findall(r'#\s*include\s*[<"]([^>"]+)[>"]', home.read_text()),
                         ["../model/flashnext/geometry.h"])

    def test_loop_home_counts(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, {LOOP_HOME: "template <class B> void RunTrunk(B b) {\n"
                                    "  for (int l = 0; l < kLayers; ++l) { b(l); }\n"
                                    "  for (int l = 0; l < kLayers; ++l) Block(l);\n  Block(kLayers, x);\n}\n"})
            self.assertEqual(scan(Path(tmp)), {LOOP_HOME: {"trunk": 2, "mtp": 1}})
            self.assertEqual(loop_home_errors(Path(tmp)),
                             [f"{LOOP_HOME}: 2 trunk loops, want 1", f"{LOOP_HOME}: 1 MTP driver, want 0"])
            _write(tmp, {LOOP_HOME: "template <class B> void RunTrunk(B b) {\n"
                                    "  for (int l = 0; l < kLayers; ++l) b(l);\n}\n"})
            self.assertEqual(loop_home_errors(Path(tmp)), [])
            _write(tmp, {LOOP_HOME: "void Other() { for (int l = 0; l < kLayers; ++l) f(l); }\n"})
            self.assertEqual(loop_home_errors(Path(tmp)), [f"{LOOP_HOME}: no RunTrunk"])

    def test_messages_name_the_rule(self):
        self.assertIn("engine/model/", ADVICE["walker"])
        self.assertIn("RunTrunk", ADVICE["trunk"])

if __name__ == "__main__":
    if sys.argv[1:] == []:
        found = scan()
        for path, counts in found.items():
            print(f"    {path!r}: {counts},")
        print(f"# {totals(found)}")
    else:
        unittest.main()
