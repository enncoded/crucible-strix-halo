"""Stage 1 check: torch MTP forward against the engine's draft IDs.

  python -m training.mtp_tune.check CKPT DUMPDIR NAME [NAME ...] [--depth 4]

DUMPDIR holds mtp_dump --check 1 output (NAME.f32, NAME.ids, NAME.mtp).
Reports top-1 agreement on catch-up rows (full head) and on chain drafts, and
the engine top-1/top-2 margin at each disagreement.
"""
import argparse
import time

import numpy as np
import torch

from .mtp_model import Ckpt, Mtp, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("dump")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--expert-dtype", default="float32")
    a = ap.parse_args()
    t0 = time.time()
    ck = Ckpt(a.ckpt)
    m = Mtp(ck, None, expert_dtype=getattr(torch, a.expert_dtype), full_dtype=torch.float32)
    torch.cuda.synchronize()
    print(f"load {time.time() - t0:.1f} s, mem {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    tot = agree = 0
    ctot = [0] * a.depth
    cagree = [0] * a.depth
    margins = []
    for name in a.names:
        ids = [int(v) for v in open(f"{a.dump}/{name}.ids").read().strip().split(",")]
        n = len(ids)
        st = torch.from_numpy(np.fromfile(f"{a.dump}/{name}.f32", dtype=np.float32).reshape(n, W)).cuda()
        rows, chains = {}, {}
        for line in open(f"{a.dump}/{name}.mtp"):
            f = line.split()
            if f[0] == "p":
                rows[int(f[1])] = (int(f[2]), float(f[3]), float(f[4]))
            else:
                chains[int(f[1])] = [int(v) for v in f[2:]]
        D = a.depth
        T = n - D
        idt = torch.tensor(ids, device="cuda")
        toks = [idt[k:T + k].clone() for k in range(1, D + 1)]
        for P, d in chains.items():
            if P < T:
                for k in range(1, D):
                    toks[k][P] = d[k - 1]
        t1 = time.time()
        with torch.no_grad():
            out = m.chain(st, idt, D, toks=toks)
            lv = [m.logits(h, full=True).argmax(-1).cpu() for h in out]
        torch.cuda.synchronize()
        print(f"{name}: forward {time.time() - t1:.2f} s for {T} rows x {D} levels", flush=True)
        for t in range(T):
            eid, l1, l2 = rows[t]
            tot += 1
            if lv[0][t].item() == eid:
                agree += 1
            else:
                margins.append(l1 - l2)
        for P, d in chains.items():
            if P >= T:
                continue
            for k in range(min(D, len(d))):
                ctot[k] += 1
                cagree[k] += int(lv[k][P].item() == d[k])
    print(f"catch-up top-1 agreement {agree}/{tot} = {agree / tot:.4%}")
    print("engine margins at disagreements:", sorted(round(x, 3) for x in margins)[:40])
    for k in range(a.depth):
        if ctot[k]:
            print(f"chain depth {k + 1}: {cagree[k]}/{ctot[k]} = {cagree[k] / ctot[k]:.4%}")


if __name__ == "__main__":
    main()
