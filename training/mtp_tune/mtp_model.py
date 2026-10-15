"""Torch forward of the Qwen3.8-Flash-Next MTP layer (docs/FLASHNEXT_FORWARD.md 3.3).

Frozen: token embedding, LM head, trunk final mixer, routed experts. The rest
are nn.Parameters that a fine-tune can train. Attention is dense causal, which
equals QSA while the MTP context is at most 2048 positions.

Chained (training-time-test) forward: level 1 takes (H_t, x[t+1]) at position
t; level k takes (Hdraft_{k-1}[t], x[t+k]) at position t+k-1 and attends to the
level-1 keys 0..t plus the level 2..k keys of the same chain start t, as the
engine's decode chain does.
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from . import exl3_torch

D, S, W, EPS = 2560, 4, 10240, 1e-6
NH, HD, NKV = 24, 256, 2
NEXP, TOPK = 512, 10


class Ckpt:
    def __init__(self, d, sidecar=None):
        self.d = d
        self.map = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
        self.files = {}
        patch = "mtp_hyper_connection_mixer_patch.safetensors"
        if "mtp.hyper_connection_mixer.hc_norm.weight" not in self.map:
            with safe_open(f"{d}/{patch}", framework="pt") as f:
                for k in f.keys():
                    self.map[k] = patch
        if sidecar:
            # Same rule as the engine's CRUCIBLE_MTP_SIDECAR: existing mtp.* tensors only.
            path = os.path.abspath(sidecar)
            with safe_open(path, framework="pt") as f:
                for k in f.keys():
                    if not k.startswith("mtp.") or k not in self.map:
                        raise ValueError(f"sidecar tensor {k} is not an MTP checkpoint tensor")
                    self.map[k] = path

    def t(self, name):
        f = self.map[name]
        if f not in self.files:
            self.files[f] = safe_open(f if os.path.isabs(f) else f"{self.d}/{f}", framework="pt")
        return self.files[f].get_tensor(name)

    def lin(self, prefix, dtype=torch.float32, device="cuda"):
        stored = {k: self.t(f"{prefix}.{k}") for k in ("trellis", "suh", "svh", "mul1")
                  if f"{prefix}.{k}" in self.map}
        return exl3_torch.dense(stored, dtype=dtype, device=device)


def cn(x, w1, n):
    """Centered RMSNorm over groups of n features, fp32 statistics."""
    sh = x.shape
    xf = x.float().reshape(*sh[:-1], sh[-1] // n, n)
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return (y.reshape(sh) * w1).to(x.dtype)


def rope(x, pos):
    """Split-half partial RoPE on channels 0..63, theta 1e7. x [..., n, h, d], pos [n]."""
    j = torch.arange(32, device=x.device, dtype=torch.float32)
    ang = pos.float()[:, None] * (1e7 ** (-2.0 * j / 64))[None, :]
    c, s = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
    a, b = x[..., :32].float(), x[..., 32:64].float()
    return torch.cat([(a * c - b * s).to(x.dtype), (b * c + a * s).to(x.dtype), x[..., 64:]], -1)


class Hc(nn.Module):
    def __init__(self, ck, p, inject):
        super().__init__()
        f = lambda n: nn.Parameter(ck.t(p + n).float().cuda())
        self.norm = f("hc_norm.weight")
        self.down = f("input_mix_weight_down.weight")
        self.up = f("input_mix_weight_up.weight")
        self.inject = f("block_inject_weight.weight") if inject else None

    def forward(self, H):
        """H [n, 10240] raw streams -> x [n, 2560], post [n, 4] or None."""
        N = cn(H, self.norm + 1, D)
        a = F.silu((N @ self.down.t().to(N.dtype)) / S)
        b = torch.sigmoid(a @ self.up.t().to(N.dtype))
        x = (b * N).reshape(-1, S, D).mean(1)
        post = 2 * torch.sigmoid((N @ self.inject.t().to(N.dtype)) / S) if self.inject is not None else None
        return x, post


class Mtp(nn.Module):
    def __init__(self, ck: Ckpt, vocab_ids=None, expert_dtype=torch.bfloat16, dense_dtype=torch.float32,
                 full_dtype=torch.bfloat16):
        super().__init__()
        P = "mtp.layers.0."
        A, M = P + "self_attn.", P + "mlp."
        g = lambda n: nn.Parameter(ck.t(n).float().cuda())
        self.ne = g("mtp.pre_fc_norm_embedding.weight")
        self.nh = g("mtp.pre_fc_norm_hidden.weight")
        lin = lambda n: nn.Parameter(ck.lin(n, dense_dtype))
        self.fe, self.fh = lin("mtp.fc_embedding"), lin("mtp.fc_hidden")
        self.hca = Hc(ck, P + "attn_hyper_connection.", True)
        self.hcm = Hc(ck, P + "mlp_hyper_connection.", True)
        self.mixer = Hc(ck, "mtp.hyper_connection_mixer.", False)
        self.wq, self.wk, self.wv, self.wo = (lin(A + n) for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
        self.qn, self.kn = g(A + "q_norm.weight"), g(A + "k_norm.weight")
        self.router = g(M + "gate.weight")
        self.sgate = g(M + "shared_expert_gate.weight")
        self.sg, self.su, self.sd = (lin(M + "shared_expert." + n) for n in ("gate_proj", "up_proj", "down_proj"))
        # Frozen pieces: buffers, never optimized.
        self.register_buffer("embed", ck.t("model.language_model.embed_tokens.weight").cuda(), persistent=False)
        self.final = Hc(ck, "model.language_model.hyper_connection_mixer.", False)
        for p in self.final.parameters():
            p.requires_grad_(False)
        head = ck.lin("lm_head", torch.float32)
        self.vocab_ids = vocab_ids
        self.register_buffer("head_full", head.to(full_dtype), persistent=False)
        self.register_buffer("head", (head[:, vocab_ids] if vocab_ids is not None else head).to(dense_dtype),
                             persistent=False)
        del head
        self.experts = self._experts(ck, M, expert_dtype)

    @staticmethod
    def _experts(ck, M, dtype):
        gu = torch.empty(NEXP, 2, D, 640, dtype=dtype, device="cuda")
        dn = torch.empty(NEXP, 640, D, dtype=dtype, device="cuda")
        for e in range(NEXP):
            q = f"{M}experts.{e}."
            gu[e, 0] = ck.lin(q + "gate_proj", dtype)
            gu[e, 1] = ck.lin(q + "up_proj", dtype)
            dn[e] = ck.lin(q + "down_proj", dtype)
        return gu, dn

    def trainable(self, scope="plain"):
        """plain: HC mixers, norms, router (no EXL3 tensors); all: plus EXL3 dense ones."""
        names = ["ne", "nh", "qn", "kn", "router", "sgate", "hca.", "hcm.", "mixer."]
        if scope == "all":
            names += ["fe", "fh", "wq", "wk", "wv", "wo", "sg", "su", "sd"]
        out = []
        for n, p in self.named_parameters():
            on = any(n == k or (k.endswith(".") and n.startswith(k)) for k in names) and not n.startswith("final.")
            p.requires_grad_(on)
            if on:
                out.append((n, p))
        return out

    # ---- pieces ----
    cap = None  # {name: [sum x^T x fp32, rows]} while capturing EXL3 requantization Hessians

    def mm(self, name, x):
        """x @ W for a dense EXL3-backed matrix; records the input Hessian when capturing."""
        if self.cap is not None:
            xf = x.detach().float().reshape(-1, x.shape[-1])
            h = self.cap.setdefault(name, [torch.zeros(xf.shape[1], xf.shape[1], device=x.device), 0])
            h[0].addmm_(xf.t(), xf)
            h[1] += xf.shape[0]
        return x @ getattr(self, name).to(x.dtype)

    def fuse(self, H, tok):
        """H [n, 10240] stacks, tok [n] next tokens -> F [n, 10240]."""
        E = cn(self.embed[tok].float(), self.ne + 1, D).to(H.dtype)
        Nn = cn(H, self.nh + 1, W)
        Ep = self.mm("fe", E)
        Hp = self.mm("fh", Nn.reshape(-1, D)).reshape(-1, S, D)
        return (Hp + Ep[:, None, :]).reshape(-1, W)

    def qkv(self, x, pos):
        n = x.shape[0]
        q = self.mm("wq", x).reshape(n, NH, 2 * HD)
        qq, gate = q[..., :HD], q[..., HD:]
        k = self.mm("wk", x).reshape(n, NKV, HD)
        v = self.mm("wv", x).reshape(n, NKV, HD)
        qq = rope(cn(qq, self.qn + 1, HD), pos)
        k = rope(cn(k, self.kn + 1, HD), pos)
        return qq, gate, k, v

    def moe(self, x):
        n = x.shape[0]
        lg = x.float() @ self.router.t()
        topv, topi = lg.topk(TOPK, dim=-1)
        w = torch.softmax(topv, -1)
        flat = topi.reshape(-1)
        order = torch.argsort(flat, stable=True)
        rows = order // TOPK
        counts = torch.bincount(flat, minlength=NEXP).tolist()
        gu, dn = self.experts
        xs = x[rows].to(gu.dtype)
        outs = []
        s = 0
        for e, c in enumerate(counts):
            if c == 0:
                continue
            xe = xs[s:s + c]
            h = xe @ gu[e, 0]
            u = xe @ gu[e, 1]
            outs.append((F.silu(h) * u) @ dn[e])
            s += c
        y_sorted = torch.cat(outs).to(x.dtype) * w.reshape(-1)[order][:, None].to(x.dtype)
        y = torch.zeros(n, D, dtype=x.dtype, device=x.device).index_add(0, rows, y_sorted)
        sh = torch.sigmoid(x.float() @ self.sgate.t()).to(x.dtype)
        hs = F.silu(self.mm("sg", x)) * self.mm("su", x)
        return y + sh * self.mm("sd", hs), topi

    def logits(self, H, full=False):
        x, _ = self.mixer(H)
        return x @ (self.head_full if full else self.head).to(x.dtype)

    @torch.no_grad()
    def teacher(self, H, full=False):
        x, _ = self.final(H)
        return x @ (self.head_full if full else self.head).to(x.dtype)

    # ---- chained forward ----
    def chain(self, stacks, ids, depth, pos0=0, toks=None):
        """stacks [T+depth, 10240] trunk stacks, ids [T+depth] tokens.

        toks: optional per-level input tokens [T] (default ids[k:T+k]).

        Returns a list of draft stacks per level, each [T, 10240]: level k row t
        predicts token t+k+1 (teacher row t+k).
        """
        T = stacks.shape[0] - depth
        dev = stacks.device
        t_idx = torch.arange(T, device=dev)
        keys, vals = [], []
        out = []
        H = stacks[:T]
        for k in range(1, depth + 1):
            Fs = self.fuse(H, ids[k:T + k] if toks is None else toks[k - 1])
            x, post = self.hca(Fs)
            pos = pos0 + t_idx + (k - 1)
            q, gate, kk, vv = self.qkv(x, pos)
            keys.append(kk)
            vals.append(vv)
            ctx = self.attend(q, keys, vals)
            y = self.mm("wo", (ctx * torch.sigmoid(gate)).reshape(T, NH * HD))
            Fs = Fs + (post[:, :, None] * y[:, None, :]).reshape(T, W)
            x, post = self.hcm(Fs)
            y, _ = self.moe(x)
            Fs = Fs + (post[:, :, None] * y[:, None, :]).reshape(T, W)
            out.append(Fs)
            H = Fs
        return out

    @staticmethod
    def attend(q, keys, vals):
        """q [T, 24, 256]; keys[0] causal over T, keys[1:] diagonal (same chain start)."""
        T = q.shape[0]
        rep = NH // NKV
        qf = q.float().transpose(0, 1)                                   # [24, T, d]
        k1 = keys[0].float().repeat_interleave(rep, 1).transpose(0, 1)   # [24, T, d]
        v1 = vals[0].float().repeat_interleave(rep, 1).transpose(0, 1)
        base = qf @ k1.transpose(1, 2) / 16.0                            # [24, T, T]
        mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        base = base.masked_fill(~mask, float("-inf"))
        extra = [(qf * kj.float().repeat_interleave(rep, 1).transpose(0, 1)).sum(-1) / 16.0
                 for kj in keys[1:]]                                     # each [24, T]
        allg = torch.cat([base] + [e[..., None] for e in extra], -1)
        p = torch.softmax(allg, -1)
        ctx = p[..., :T] @ v1
        for j, vj in enumerate(vals[1:]):
            ctx = ctx + p[..., T + j:T + j + 1] * vj.float().repeat_interleave(rep, 1).transpose(0, 1)
        return ctx.transpose(0, 1).to(q.dtype)                           # [T, 24, 256]


def load_blocks(path):
    blocks = [int(v) for v in open(path).read().replace("\n", ",").split(",") if v.strip()]
    return torch.tensor([b * 128 + c for b in blocks for c in range(128)], dtype=torch.long)
