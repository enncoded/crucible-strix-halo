"""Compare next-token logits: KL(test || base), reverse KL and top-1. Usage: pl_kl.py BASE TEST"""
import sys
import numpy as np

def logp(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())

base, test = (np.fromfile(p, dtype=np.float32) for p in sys.argv[1:3])
lb, lt = logp(base), logp(test)
pb, pt = np.exp(lb), np.exp(lt)
print(f"KL(test||base)={np.sum(pt*(lt-lb)):.8f} KL(base||test)={np.sum(pb*(lb-lt)):.8f} "
      f"top1 base={base.argmax()} test={test.argmax()}")
