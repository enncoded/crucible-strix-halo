"""The work layout of the tools, in one place.

The tools keep the GPU queue program, the reference environment, the reference prompt and the gate cache
under one root. Set WORK_DIR to move the root; the default is ~/crucible-work. Other files use this module
(or tools/work_paths.sh) and never name the root.

WORK_DIR has no CRUCIBLE_ prefix on purpose: the suite and the gates refuse unknown CRUCIBLE_ names.
The queue program is $WORK_DIR/bin/gpu-run. tools/gpu-run is a simple stand-in: copy it there, or put your own
queue program there with the same command line (`gpu-run [-p PRIO] [-w SECONDS] -- cmd [args...]`).
"""
import os
from pathlib import Path

ENV = "WORK_DIR"


def root():
    value = os.environ.get(ENV)
    return Path(value) if value else Path.home() / "crucible-work"


def gpu_run():
    return root() / "bin/gpu-run"


def refenv_python():
    return root() / "refenv/.venv/bin/python"


def fibonacci_ids():
    return root() / "refenv/out/fibonacci/ids.txt"


def worktree(name):
    return root() / "wt" / name


def gates_cache():
    return root() / "cache/gates"
