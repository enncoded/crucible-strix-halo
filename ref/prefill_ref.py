"""Literal fp64 prefill algebra, deliberately independent of HIP indexing."""
import numpy as np


def serial_gdn(q, k, v, g, beta, initial):
    state = initial.copy()
    out = []
    for qt, kt, vt, gt, bt in zip(q, k, v, g, beta):
        decayed = np.exp(gt) * state
        delta = bt * (vt - kt @ decayed)
        state = decayed + np.outer(kt, delta)
        out.append(qt @ state)
    return np.asarray(out), state


def chunk_gdn(q, k, v, g, beta, initial, chunk=64):
    state = initial.copy()
    out = []
    for start in range(0, len(q), chunk):
        qc, kc, vc = q[start:start + chunk], k[start:start + chunk], v[start:start + chunk]
        bc = beta[start:start + chunk]
        cumulative = np.cumsum(g[start:start + chunk])
        # No reciprocal product: decay may safely underflow, never overflow.
        decay = np.exp(np.minimum(cumulative[:, None] - cumulative[None, :], 0))
        lower = np.tril(bc[:, None] * (kc @ kc.T) * decay, -1)
        inverse = np.linalg.solve(np.eye(len(qc)) + lower, np.eye(len(qc)))
        ut = inverse @ (bc[:, None] * vc)
        w = inverse @ (bc[:, None] * np.exp(cumulative[:, None]) * kc)
        u = ut - w @ state
        local = np.tril((qc @ kc.T) * decay) @ u
        out.append(np.exp(cumulative[:, None]) * (qc @ state) + local)
        end_keys = kc * np.exp(cumulative[-1] - cumulative[:, None])
        state = np.exp(cumulative[-1]) * state + end_keys.T @ u
    return np.concatenate(out), state


def dispatch(selections, experts=512):
    selected = np.asarray(selections)
    if selected.ndim != 2 or np.any(selected < 0) or np.any(selected >= experts):
        raise ValueError("expert IDs must be in range")
    flat = selected.ravel()
    counts = np.bincount(flat, minlength=experts)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    pairs = np.argsort(flat, kind="stable")
    return counts, offsets, pairs


def selected_tokens(scores, n):
    nb = n // 4
    if nb <= 512:
        return np.arange(n)
    blocks = np.array(sorted(range(nb), key=lambda i: (-scores[i], i))[:512])
    blocks.sort()
    return np.concatenate(((blocks[:, None] * 4 + np.arange(4)).ravel(), np.arange(4 * nb, n)))
