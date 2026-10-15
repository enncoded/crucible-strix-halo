#!/usr/bin/env python3
"""fp64 single-layer oracle for Qwen3.8-Flash-Next EXL3 (CPU, NumPy + torch I/O).

Teacher-forced: takes a dumped residual stack h_L ([n][4][2560] float32) and
computes h_{L+1} for one trunk layer in float64 from the checkpoint's exact
decoded weights (W = diag(suh) H W_hat H diag(svh), no intermediate rounding).
Routed MoE is evaluated only at --pos positions (decoding every expert of
every position is slow); the report covers those positions.

  oracle.py CKPT DUMP_DIR LAYER --pos 0,14,23 [--other DIR2]

Prints, for DUMP_DIR's own h_{L+1} (and DIR2's when given, each fed its own
h_L), the relative L2 error against the oracle at each position.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(__file__))
import exl3_ref  # noqa: E402

D, S, W = 2560, 4, 10240
EPS = 1e-6


class Ckpt:
    def __init__(self, d):
        self.d = d
        self.map = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
        self.files = {}
        self.cache = {}
        extra = "ngram_embedding.safetensors"
        self.files[extra] = safe_open(f"{d}/{extra}", framework="pt")
        for k in self.files[extra].keys():
            self.map[k] = extra

    def t(self, name):
        f = self.map[name]
        if f not in self.files:
            self.files[f] = safe_open(f"{self.d}/{f}", framework="pt")
        return self.files[f].get_tensor(name)

    def f64(self, name):
        return self.t(name).to(torch.float64).numpy()

    def lin(self, prefix):
        if prefix not in self.cache:
            stored = {k: self.t(f"{prefix}.{k}").numpy() for k in ("trellis", "suh", "svh", "mul1")}
            what = exl3_ref.reconstruct_inner(stored).astype(np.float64)
            self.cache[prefix] = (what, stored["suh"].astype(np.float64), stored["svh"].astype(np.float64))
        return self.cache[prefix]


H128 = None


def had(x):
    """Normalized natural-order Sylvester transform over consecutive 128-blocks."""
    global H128
    if H128 is None:
        idx = np.arange(128)
        pc = np.vectorize(lambda v: bin(v).count("1"))(idx[:, None] & idx[None, :])
        H128 = (1.0 - 2.0 * (pc & 1)) / np.sqrt(128.0)
    sh = x.shape
    return (x.reshape(*sh[:-1], sh[-1] // 128, 128) @ H128).reshape(sh)


def linear(ck, prefix, x):
    what, suh, svh = ck.lin(prefix)
    return had(had(x * suh) @ what) * svh


def rms(x, w1):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w1


def silu(x):
    return x / (1 + np.exp(-x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hc_read(ck, p, Hs, inject=True):
    w1 = ck.f64(p + "hc_norm.weight").reshape(S, D) + 1.0
    N = rms(Hs, w1)                                     # [n,4,D]
    f = N.reshape(len(Hs), W)
    a = silu(f @ ck.f64(p + "input_mix_weight_down.weight").T / S)
    b = sigmoid(a @ ck.f64(p + "input_mix_weight_up.weight").T).reshape(-1, S, D)
    x = (b * N).mean(1)
    post = 2 * sigmoid(f @ ck.f64(p + "block_inject_weight.weight").T / S) if inject else None
    return x, post


def gdn(ck, p, x):
    n = len(x)
    qkv = linear(ck, p + "in_proj_qkv", x)
    z = linear(ck, p + "in_proj_z", x).reshape(n, 48, 128)
    a = x @ ck.f64(p + "in_proj_a.weight").T
    b = x @ ck.f64(p + "in_proj_b.weight").T
    cw = ck.f64(p + "conv1d.weight")[:, 0, :]           # [10240,4]
    pad = np.concatenate([np.zeros((3, W)), qkv])
    c = silu(sum(cw[:, t] * pad[t:t + n] for t in range(4)))
    q = c[:, :2048].reshape(n, 16, 128)
    k = c[:, 2048:4096].reshape(n, 16, 128)
    v = c[:, 4096:].reshape(n, 48, 128)
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + EPS) / np.sqrt(128)
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + EPS)
    q, k = np.repeat(q, 3, axis=1), np.repeat(k, 3, axis=1)
    A = -np.exp(ck.f64(p + "A_log"))
    sp = a + ck.f64(p + "dt_bias")
    g = A * np.where(sp > 20, sp, np.log1p(np.exp(np.minimum(sp, 20))))
    decay, beta = np.exp(g), sigmoid(b)
    st = np.zeros((48, 128, 128))                       # [h, key, value]
    y = np.zeros((n, 48, 128))
    for t in range(n):
        st = st * decay[t][:, None, None]
        mem = np.einsum("hij,hi->hj", st, k[t])
        delta = beta[t][:, None] * (v[t] - mem)
        st = st + k[t][:, :, None] * delta[:, None, :]
        y[t] = np.einsum("hij,hi->hj", st, q[t])
    gamma = ck.f64(p + "norm.weight")
    o = y / np.sqrt((y * y).mean(-1, keepdims=True) + EPS) * gamma * sigmoid(z)
    return linear(ck, p + "out_proj", o.reshape(n, 6144))


def rope(x, pos):
    """Split-half partial RoPE on channels 0..63, theta 1e7, text positions."""
    j = np.arange(32)
    ang = pos[:, None] * (1e7 ** (-2.0 * j / 64))[None, :]
    c, s = np.cos(ang), np.sin(ang)
    while c.ndim < x.ndim:
        c, s = c[:, None], s[:, None]
    out = x.copy()
    a, b = x[..., :32], x[..., 32:64]
    out[..., :32] = a * c - b * s
    out[..., 32:64] = b * c + a * s
    return out


def qsa(ck, p, x):
    """Dense causal attention (exact for contexts of at most 2048 tokens)."""
    n = len(x)
    pos = np.arange(n, dtype=np.float64)
    q = linear(ck, p + "q_proj", x).reshape(n, 24, 512)
    qq, gate = q[..., :256], q[..., 256:]
    k = linear(ck, p + "k_proj", x).reshape(n, 2, 256)
    v = linear(ck, p + "v_proj", x).reshape(n, 2, 256)
    qq = rope(rms(qq, ck.f64(p + "q_norm.weight") + 1), pos)
    k = rope(rms(k, ck.f64(p + "k_norm.weight") + 1), pos)
    ctx = np.zeros((n, 24, 256))
    for h in range(24):
        g = h // 12
        lg = qq[:, h] @ k[:, g].T / 16.0
        lg = np.where(np.tril(np.ones((n, n))) > 0, lg, -np.inf)
        pr = np.exp(lg - lg.max(1, keepdims=True))
        pr /= pr.sum(1, keepdims=True)
        ctx[:, h] = pr @ v[:, g]
    return linear(ck, p + "o_proj", (ctx * sigmoid(gate)).reshape(n, 6144))


def ple_rows(ck, ids):
    """Hashed n-gram rows (16 per position); only 248044 resets history."""
    N = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
    mult = ck.t(N + "layer_multipliers").numpy().astype(np.uint64)
    voc = ck.t(N + "head_vocab_sizes").numpy().astype(np.uint64)
    off = ck.t(N + "head_offsets").numpy()
    rows = []
    for t in range(len(ids)):
        e = max([i for i in range(t) if ids[i] == 248044], default=-1)
        u = [np.uint64(ids[t - k]) if t - k > e and t - k >= 0 else np.uint64(248044) for k in range(3)]
        with np.errstate(over="ignore"):
            h2 = (u[0] * mult[0]) ^ (u[1] * mult[1])
            h3 = h2 ^ (u[2] * mult[2])
        rows.append([int((h2 if h < 8 else h3) % voc[h]) + int(off[h]) for h in range(16)])
    return rows


def ple_embed(ck, ids):
    N = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
    bias = ck.f64(N + "head_bias")
    unsharded = N + "trellis" in ck.map
    shard_rows = None if unsharded else ck.t(N + "shard_0.trellis").shape[0]
    out = np.zeros((len(ids), D))
    for t, rr in enumerate(ple_rows(ck, ids)):
        for h, row in enumerate(rr):
            table = ck.t(N + "trellis") if unsharded else ck.t(f"{N}shard_{row // shard_rows}.trellis")
            pk = table[row if unsharded else row % shard_rows].numpy().view(np.uint16)
            K, rem = divmod((len(pk) - 1) * 16, 160)
            if rem or K not in (5, 6):
                raise ValueError("Unsupported PLE row codec")
            scale = pk[0:1].view(np.float16).astype(np.float64)[0]
            bits = ((pk[1:].astype(np.uint32)[:, None] >> np.arange(16)) & 1).reshape(-1)
            i = np.arange(160)[:, None]
            m = np.arange(16)[None, :]
            st = (bits[((i - m // K) % 160) * K + m % K].astype(np.uint32) << m).sum(1)
            val = exl3_ref.decode_codebook(st.astype(np.uint16)).astype(np.float64)
            out[t, h * 160:(h + 1) * 160] = val * scale + bias[h]
    return out


def ple(ck, Hs, ids):
    p = "model.language_model.layers.1.ple."
    n = len(Hs)
    e = ple_embed(ck, ids)
    key = (e @ ck.f64(p + "key_proj.weight").T).reshape(n, S, D)
    val = e @ ck.f64(p + "value_proj.weight").T
    key = rms(key, ck.f64(p + "norm_key.weight").reshape(S, D) + 1)
    qry = rms(Hs, ck.f64(p + "norm_query.weight").reshape(S, D) + 1)
    sc = (key * qry).sum(-1) / np.sqrt(D)
    a = np.sign(sc) * np.sqrt(np.maximum(np.abs(sc), 1e-6))
    U = sigmoid(a)[:, :, None] * val[:, None, :]
    Nn = rms(U, ck.f64(p + "norm_conv.weight").reshape(S, D) + 1).reshape(n, W)
    cw = ck.f64(p + "conv1d.weight")[:, 0, :]
    pad = np.concatenate([np.zeros((9, W)), Nn])
    conv = sum(cw[:, j] * pad[3 * j:3 * j + n] for j in range(4))
    return Hs + U + silu(conv).reshape(n, S, D)


def moe(ck, p, x):
    lg = x @ ck.f64(p + "gate.weight").T
    top = sorted(range(512), key=lambda e: (-lg[e], e))[:10]
    e = np.exp(lg[top] - lg[top].max())
    w = e / e.sum()
    y = np.zeros(D)
    for wk, ei in zip(w, top):
        q = f"{p}experts.{ei}."
        h = silu(linear(ck, q + "gate_proj", x)) * linear(ck, q + "up_proj", x)
        y += wk * linear(ck, q + "down_proj", h)
    sg = sigmoid(x @ ck.f64(p + "shared_expert_gate.weight").T)[0]
    q = p + "shared_expert."
    h = silu(linear(ck, q + "gate_proj", x)) * linear(ck, q + "up_proj", x)
    return y + sg * linear(ck, q + "down_proj", h), top


def layer(ck, L, Hin, pos, ids):
    P = f"model.language_model.layers.{L}."
    Hs = Hin.astype(np.float64).copy()
    if L == 1:
        Hs = ple(ck, Hs, ids)
    x, post = hc_read(ck, P + "attn_hyper_connection.", Hs)
    y = qsa(ck, P + "self_attn.", x) if L % 4 == 3 else gdn(ck, P + "linear_attn.", x)
    Hs = Hs + post[:, :, None] * y[:, None, :]
    out, tops = {}, {}
    for t in pos:
        h1 = Hs[t:t + 1]
        x, post = hc_read(ck, P + "mlp_hyper_connection.", h1)
        y, tops[t] = moe(ck, P + "mlp.", x[0])
        out[t] = h1[0] + post[0][:, None] * y[None, :]
    return out, tops


def load(d, i):
    return np.fromfile(f"{d}/h_{i:02d}.f32", dtype=np.float32).reshape(-1, S, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("dump")
    ap.add_argument("layer", type=int)
    ap.add_argument("--pos", default="0")
    ap.add_argument("--other")
    ap.add_argument("--ids", required=True, help="file with comma-separated prompt ids")
    a = ap.parse_args()
    ck = Ckpt(a.ckpt)
    pos = [int(v) for v in a.pos.split(",")]
    ids = [int(v) for v in open(a.ids).read().strip().split(",")]
    for d in [a.dump] + ([a.other] if a.other else []):
        hin, hout = load(d, a.layer), load(d, a.layer + 1)
        n = min(len(hin), len(hout), len(ids))
        oracle, tops = layer(ck, a.layer, hin[:n], pos, ids[:n])
        for t in pos:
            ref = oracle[t]
            err = np.linalg.norm(hout[t] - ref) / np.linalg.norm(ref)
            dlt = np.linalg.norm(ref - hin[t]) / np.linalg.norm(ref)
            print(f"{d}: layer {a.layer} pos {t}: relL2 vs fp64 {err:.3e} "
                  f"(layer update {dlt:.3e} of |h|), experts {tops[t]}")


if __name__ == "__main__":
    main()
