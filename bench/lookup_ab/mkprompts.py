#!/usr/bin/env python3
"""Two closed-turn 16K prompts for the lookup ABAB (public text only).

copy:    the mtp_tune family c prompt (a CPython function far back, then a request to repeat it verbatim).
summary: the kv-quant 16K CPython tail with a request to summarize it; the reply is prose, not a copy.
Usage: mkprompts.py DEPTH_P_DIR KVQ_P_DIR TOKENIZER_JSON OUT_DIR
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path
from tokenizers import Tokenizer

ASK = "\n\nSummarize what this code does in a few short paragraphs."
REPLY = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main():
    depth_p, kvq_p, tokenizer, out = map(Path, sys.argv[1:5])
    tok = Tokenizer.from_file(str(tokenizer))
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(depth_p / "c_16384.txt", out / "copy.txt")
    tail = [int(x) for x in (kvq_p / "py_16384.txt").read_text().split(",")]
    end = tok.encode(ASK, add_special_tokens=False).ids + tok.encode(REPLY, add_special_tokens=False).ids
    if 248046 not in end:
        raise SystemExit("the reply template did not encode to special tokens")
    summary = tail[:16384 - len(end)] + end
    (out / "summary.txt").write_text(",".join(map(str, summary)))
    print(json.dumps({name: hashlib.sha256((out / f"{name}.txt").read_bytes()).hexdigest()
                      for name in ("copy", "summary")}))


if __name__ == "__main__":
    main()
