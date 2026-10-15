"""Training-step speed and memory on gfx1151 (stage 1 feasibility).

  python -m training.mtp_tune.bench_train CKPT DUMPDIR NAME [--T 1024] [--depth 4]
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from .mtp_model import Ckpt, Mtp, W, load_blocks


def step(m, st, idt, T, D, opt):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            tl = m.teacher(st[1:T + D]).float()                      # rows 1..T+D-1
        out = m.chain(st[:T + D], idt[:T + D], D)
        loss = 0
        for k, h in enumerate(out):
            lg = m.logits(h).float()
            tgt = torch.softmax(tl[k:k + T], -1)
            loss = loss + F.kl_div(torch.log_softmax(lg, -1), tgt, reduction="batchmean")
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("dump")
    ap.add_argument("name")
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--blocks", default="tests/data/draft/blocks65536.txt")
    a = ap.parse_args()
    t0 = time.time()
    m = Mtp(Ckpt(a.ckpt), load_blocks(a.blocks).cuda())
    print(f"load {time.time() - t0:.1f} s, mem {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    ids = [int(v) for v in open(f"{a.dump}/{a.name}.ids").read().strip().split(",")]
    n = len(ids)
    st = torch.from_numpy(np.fromfile(f"{a.dump}/{a.name}.stk", dtype=np.float16).reshape(n, W)).cuda().float()
    idt = torch.tensor(ids, device="cuda")
    for scope in ("plain", "all"):
        params = [p for _, p in m.trainable(scope)]
        print(scope, "trainable", sum(p.numel() for p in params), flush=True)
        opt = torch.optim.AdamW(params, lr=1e-6)
        for i in range(4):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t1 = time.time()
            loss = step(m, st, idt, a.T, a.depth, opt)
            torch.cuda.synchronize()
            print(f"  step {i}: {time.time() - t1:.2f} s, loss {loss:.4f}, "
                  f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        del opt


if __name__ == "__main__":
    main()
