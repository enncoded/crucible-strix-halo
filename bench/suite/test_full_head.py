from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class FullHeadPolicyTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "g++ is needed for the CPU policy test")
    def test_old_and_new_option_structures(self):
        source = r'''
#include <cassert>
#include <string>
#include "full_head_policy.h"
struct Old {
    std::string draft_blocks = "sparse";
    bool draft_dense = true;
    int adapt = 1, lookup = 1, selection_options = 1, last_head = 0, overlap = 1;
    float draft_q = .3f, draft_margin = 1.f;
};
struct New : Old {
    std::string draft_head = "64k";
    bool adapt_set = false;
    int head_auto = 1;
};
int main() {
    Old old;
    New next;
    SetFullHeadPolicy(old);
    SetFullHeadPolicy(next);
    assert(old.draft_blocks.empty() && !old.draft_dense && !old.adapt);
    assert(!old.draft_q && !old.draft_margin && !old.lookup && !old.overlap);
    assert(!old.selection_options && old.last_head == 1);
    assert(next.draft_head == "full" && next.adapt_set && !next.head_auto);
    assert(!next.draft_q && next.draft_blocks.empty());
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "test.cpp").write_text(source)
            subprocess.run(["g++", "-std=c++20", "-I", str(Path(__file__).resolve().parent),
                            str(path / "test.cpp"), "-o", str(path / "test")], check=True)
            subprocess.run([str(path / "test")], check=True)


if __name__ == "__main__":
    unittest.main()
