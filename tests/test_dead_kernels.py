"""Ratchet: every kernel in engine/ has a launch site in engine/.

A `__global__` function in engine/ counts as launched when engine/ code names it outside its own
definition: `K<<<...>>>`, `K<T><<<...>>>`, `hipLaunchKernelGGL(K, ...)`, or a kernel pointer
(`auto k = c ? K<1> : K<2>;`). Comments and string literals do not count. A kernel that only a
bench launches belongs in bench/kernels/; a kernel with no user is deleted.
ALLOWED lists the kernels with no engine launch. It may only get shorter: the test fails on a new
dead kernel and on a stale entry. 31 kernels on train 19 (c17a47f1); 0 after an earlier change.
Run `python3 tests/test_dead_kernels.py` to print the current list.
"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_model_geometry import blank_lexemes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {".h", ".hip", ".inc", ".cc", ".cpp"}
GLOBAL = re.compile(r"\b__global__\b")
# Attributes between `__global__` and the kernel name: `__launch_bounds__(256)`,
# `__attribute__((amdgpu_waves_per_eu(8, 8)))`. Object-like macros (no parentheses) stay.
ATTRIBUTE = re.compile(r"\b__launch_bounds__\s*\((?:[^()]|\([^()]*\))*\)|"
                       r"\b__attribute__\s*\(\((?:[^()]|\([^()]*\))*\)\)")
NAME = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")


def definitions(code):
    """Yield (name, offset of the name) for each `__global__` definition or declaration in code."""
    for m in GLOBAL.finditer(code):
        ends = [i for i in (code.find("{", m.end()), code.find(";", m.end())) if i >= 0]
        head = code[m.end():min(ends, default=len(code))]
        head = ATTRIBUTE.sub(lambda a: " " * len(a.group()), head)
        name = NAME.search(head)
        if name:
            yield name.group(1), m.end() + name.start(1)


def scan(root=ROOT):
    """Return {kernel: [path, ...]} for each kernel in engine/ that engine/ code never names."""
    sources = {}
    for path in sorted((root / "engine").rglob("*")):
        if path.suffix in SUFFIXES and path.is_file():
            # Strings blank to "" so a kernel name in a message is no launch; offsets stay per file.
            sources[path.relative_to(root).as_posix()] = blank_lexemes(path.read_text())[1]
    defined = {}
    for rel, code in sources.items():
        for name, at in definitions(code):
            defined.setdefault(name, set()).add((rel, at))
    launched = set()
    for rel, code in sources.items():
        for m in IDENTIFIER.finditer(code):
            if m.group() in defined and (rel, m.start()) not in defined[m.group()]:
                launched.add(m.group())
    return {name: sorted({rel for rel, _ in sites}) for name, sites in sorted(defined.items())
            if name not in launched}


# Frozen after an earlier change (train 20): no engine kernel lacks a launch.
ALLOWED = {}


def scan_text(files):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for rel, text in files.items():
            Path(tmp, rel).parent.mkdir(parents=True, exist_ok=True)
            Path(tmp, rel).write_text(text)
        return scan(Path(tmp))


class DeadKernelRatchet(unittest.TestCase):
    def test_every_engine_kernel_has_an_engine_launch(self):
        new = {name: paths for name, paths in scan().items() if name not in ALLOWED}
        self.assertEqual(new, {}, "launch these kernels in engine/, move them to bench/kernels/, or delete them")

    def test_allow_list_has_no_stale_entry(self):
        found = scan()
        self.assertEqual(sorted(set(ALLOWED) - set(found)), [], "remove these ALLOWED entries")

    def test_scan_reads_every_engine_kernel(self):
        # A parser fault would find no kernel and pass the ratchet: each `__global__` must yield a name.
        # 124 kernels on train 20.
        codes = [blank_lexemes(p.read_text())[1] for p in sorted((ROOT / "engine").rglob("*"))
                 if p.suffix in SUFFIXES and p.is_file()]
        names = [name for code in codes for name, _ in definitions(code)]
        self.assertEqual(len(names), sum(len(GLOBAL.findall(code)) for code in codes))
        self.assertGreaterEqual(len(set(names)), 120)

    def test_launch_forms_count(self):
        self.assertEqual(scan_text({
            "engine/k.h": "__global__ void A(int x) {}\n"
                          "template <int R> __global__ void __launch_bounds__(256) B(int x) {}\n"
                          "__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8, 8))) MODE\n"
                          "C(int x) {}\n__global__ void D(int x) {}\n__global__ void E(int x);\n",
            "engine/run.hip": "void Run() {\n  A<<<1, 256>>>(0);\n  B<4><<<1, 256>>>(0);\n"
                              "  auto k = true ? C : C;\n  hipLaunchKernelGGL(D, dim3(1), dim3(1), 0, 0, 0);\n}\n",
        }), {"E": ["engine/k.h"]})

    def test_comments_strings_and_benches_are_no_launch(self):
        self.assertEqual(scan_text({
            "engine/k.h": "// Run A<<<1, 256>>> here.\n__global__ void A(int x) {}\n"
                          "__global__ void B(int x) { /* B<<<1,1>>> */ }\n",
            "engine/run.hip": 'void Run() { Log("A B"); }\n',
            "bench/b.hip": "void Bench() { A<<<1, 1>>>(0); B<<<1, 1>>>(0); }\n",
        }), {"A": ["engine/k.h"], "B": ["engine/k.h"]})

    def test_a_declaration_is_no_launch(self):
        self.assertEqual(scan_text({
            "engine/k.h": "__global__ void A(int x);\n__global__ void A(int x) {}\n",
        }), {"A": ["engine/k.h"]})


if __name__ == "__main__":
    if sys.argv[1:] == []:
        dead = scan()
        for name, paths in dead.items():
            print(f"    {name!r},  # {', '.join(paths)}")
        print(f"# {len(dead)} kernels")
    else:
        unittest.main()
