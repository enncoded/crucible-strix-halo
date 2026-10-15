# The work layout of the tools (see tools/work_paths.py). Source this file.
# Set WORK_DIR to move the root; the default is ~/crucible-work.
# WORK_DIR has no CRUCIBLE_ prefix on purpose: the suite and the gates refuse unknown CRUCIBLE_ names.
: "${WORK_DIR:=$HOME/crucible-work}"
# Exported so that a Python tool that this shell starts (tools/work_paths.py) finds the same root. The WORK_
# path names below stay local to the sourcing shell.
export WORK_DIR
WORK_GPU_RUN=$WORK_DIR/bin/gpu-run
WORK_REFENV_PYTHON=$WORK_DIR/refenv/.venv/bin/python
WORK_FIB_IDS=$WORK_DIR/refenv/out/fibonacci/ids.txt
