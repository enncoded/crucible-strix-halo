#!/usr/bin/env python3
"""Tokenizer helper (runs in the refenv venv on the test machine).

  tok.py CKPT encode "text" [--chat]   -> comma-separated ids
  tok.py CKPT decode 1,2,3             -> text
"""
import sys

from tokenizers import Tokenizer


def main():
    ckpt, cmd, arg = sys.argv[1], sys.argv[2], sys.argv[3]
    tok = Tokenizer.from_file(f"{ckpt}/tokenizer.json")
    if cmd == "encode":
        text = arg
        if "--chat" in sys.argv:
            # Qwen chat format with thinking disabled (empty think block).
            text = (f"<|im_start|>user\n{arg}<|im_end|>\n<|im_start|>assistant\n"
                    "<think>\n\n</think>\n\n")
        print(",".join(str(i) for i in tok.encode(text, add_special_tokens=False).ids))
    else:
        ids = [int(x) for x in arg.split(",") if x]
        print(tok.decode(ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
