"""The symbol scan of tools/layout_symbols.py, on small fixtures."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import layout  # noqa: E402
import layout_symbols as sym  # noqa: E402


class DefinedSymbolsTest(unittest.TestCase):
    def test_top_level_names(self):
        text = '''
        #define MACRO_NAME 3
        using Alias = int;
        typedef unsigned Old;
        namespace crucible {
        struct Plain { int member; void Method() {} };
        enum class Mode { A, B };
        inline float Fn(float x) { return x; }
        constexpr float kPrior = 0.75f;
        constexpr float kEdges[] = {1, 2};
        template <class T> T Tpl(const std::string& v) { return T{}; }
        template <class T> struct Box { T value; };
        }
        bool Free(const Plain& o) { return o.member != 0; }
        struct Fwd;
        '''
        self.assertEqual(sym.defined_symbols(text), {
            'MACRO_NAME', 'Alias', 'Old', 'Plain', 'Mode', 'Fn', 'kPrior', 'kEdges', 'Tpl', 'Box', 'Free'})

    def test_members_locals_and_literals_are_not_symbols(self):
        text = '''
        // struct InComment { };
        const char* s = "struct InString { };";
        void F() { int local = 1; auto g = [&] { return local; }; (void)g; }
        class C { public: static int Count() { return 1; } int field; };
        using namespace crucible;
        extern "C" { int CFn(int); }
        '''
        names = sym.defined_symbols(text)
        self.assertEqual(names, {'s', 'F', 'C', 'CFn'})

    def test_raw_strings_and_directives_are_skipped(self):
        text = 'const char* k = R"x(struct Hidden { })x";\n#if defined(A) \\\n && B\nint After = 1;\n'
        self.assertEqual(sym.defined_symbols(text), {'k', 'After'})


class UsedIdentifiersTest(unittest.TestCase):
    def test_members_and_comments_do_not_count(self):
        text = 'int x = Foo(a.Bar, b->Baz); // Qux\nconst char* s = "Quux";'
        used = sym.used_identifiers(text)
        self.assertTrue({'Foo', 'a', 'b'} <= used)
        self.assertFalse({'Bar', 'Baz', 'Qux', 'Quux'} & used)

    def test_a_name_declared_in_the_file_is_a_local(self):
        text = 'struct S { void Hc(int p) {} };\nvoid F(S& s) { s.Hc(1); Hc(2); int Loader = 0; }'
        self.assertFalse({'Hc', 'Loader'} & sym.used_identifiers(text))

    def test_calls_of_a_name_are_uses(self):
        text = 'void F() { auto q = DraftAcceptProb(1.f); return Run(q, kPrior); }'
        self.assertTrue({'DraftAcceptProb', 'Run', 'kPrior'} <= sym.used_identifiers(text))


def tree(files):
    """Rows and file texts for an in-memory tree: {path: (module, text)}."""
    rows = tuple(layout.Row(p, p, module, '', '') for p, (module, _) in files.items())
    return layout.Layout(rows), frozenset(files)


class ViolationsTest(unittest.TestCase):
    def scan(self, files, tmp):
        for path, (_, text) in files.items():
            target = Path(tmp) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        lay, tracked = tree(files)
        return sym.violations(Path(tmp), lay, tracked)

    def run_scan(self, files):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            return self.scan(files, tmp)

    def test_use_of_a_higher_symbol_is_found_without_an_include(self):
        found = self.run_scan({
            'engine/serve/s.h': ('serve', 'void F() { auto o = CliGenOptions(); }'),
            'engine/cli/g.h': ('cli', 'struct GenOptions {};\ninline GenOptions CliGenOptions() { return {}; }'),
        })
        self.assertEqual(found, ['symbol: engine/serve/s.h uses CliGenOptions (engine/cli/g.h)'])

    def test_lower_and_same_rank_symbols_are_fine(self):
        found = self.run_scan({
            'engine/cli/c.h': ('cli', 'void F() { Policy(); Peer(); }'),
            'engine/runtime/p.h': ('runtime', 'inline int Policy() { return 1; }'),
            'engine/cli/d.h': ('cli', 'inline int Peer() { return 1; }'),
        })
        self.assertEqual(found, [])

    def test_a_name_with_a_definition_at_or_below_the_user_is_fine(self):
        found = self.run_scan({
            'engine/runtime/r.h': ('runtime', 'void F() { Now(); }'),
            'engine/gfx1151/g.h': ('gfx1151', 'inline double Now() { return 0; }'),
            'engine/cli/c.h': ('cli', 'inline double Now() { return 1; }'),
        })
        self.assertEqual(found, [])

    def test_a_leaf_symbol_is_fine_for_every_module(self):
        found = self.run_scan({
            'engine/gfx1151/g.h': ('gfx1151', 'int F() { return SwitchOn(); }'),
            'engine/switches.h': ('engine', 'inline int SwitchOn() { return 1; }'),
        })
        self.assertEqual(found, [])


if __name__ == '__main__':
    unittest.main()
