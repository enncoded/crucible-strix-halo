"""EXL3 linear tensors decoded on the GPU with torch (port of ref/exl3_ref.py).

dense(stored) returns W [in, out] so that y = x @ W equals the reference
linear had(had(x * suh) @ W_hat) * svh, with no intermediate rounding.
"""
from __future__ import annotations

import torch

MUL1 = 0x83DCD12D
MCG = 0xCBAC1FED


def _perm(device) -> torch.Tensor:
    lane = torch.arange(32)
    rows = 2 * (lane % 4)
    cols = lane // 4
    p = torch.stack([rows * 16 + cols, (rows + 1) * 16 + cols, (rows + 8) * 16 + cols,
                     (rows + 9) * 16 + cols, rows * 16 + cols + 8, (rows + 1) * 16 + cols + 8,
                     (rows + 8) * 16 + cols + 8, (rows + 9) * 16 + cols + 8], dim=-1).reshape(256)
    return p.to(device)


def _states(tiles: torch.Tensor, bits: float) -> torch.Tensor:
    """tiles [n, 16*bits] int16 -> [n, 256] int64 16-bit decoder states."""
    whole = int(bits)
    half = bits != whole
    w16 = tiles.to(torch.int64) & 0xFFFF
    words = w16[:, 0::2] | (w16[:, 1::2] << 16)
    nw = words.shape[1]
    idx = torch.arange(256, device=tiles.device)
    widths = whole + (idx % 2 if half else torch.zeros_like(idx))
    ends = torch.cumsum(widths, 0) + int(256 * bits)
    starts = ends - 16
    first = torch.div(starts, 32, rounding_mode="floor") % nw
    last = torch.div(ends - 1, 32, rounding_mode="floor")
    shift = (last + 1) * 32 - ends
    a = words[:, first]
    b = words[:, last % nw]
    # Bits shift..shift+15 of a:b; arithmetic shift only pollutes bits >= 32.
    return ((a << 32 | b) >> shift) & 0xFFFF


def _codebook(states: torch.Tensor, codebook: str) -> torch.Tensor:
    if codebook == "mul1":
        prod = (states * MUL1) & 0xFFFFFFFF
        bsum = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255)
        h = (0x6400 + bsum).to(torch.int16).view(torch.float16).to(torch.float64)
        inv = torch.tensor([0x1EEE], dtype=torch.int16).view(torch.float16).item()
        bias = torch.tensor([0xC931 - 65536], dtype=torch.int16).view(torch.float16).item()
        return (h * inv + bias).to(torch.float16)
    mul, add = (MCG, 0) if codebook == "mcg" else (89226354, 64248484)
    v = ((states * mul + add) & 0xFFFFFFFF & 0x8FFF8FFF) ^ 0x3B603B60
    lo = (v & 0xFFFF).to(torch.int32).to(torch.int16)
    hi = (v >> 16).to(torch.int32).to(torch.int16)
    # int32 -> int16 wraps values above 32767 to the same bit pattern.
    return (lo.view(torch.float16).float() + hi.view(torch.float16).float()).to(torch.float16)


def inner(trellis: torch.Tensor, codebook: str = "mul1") -> torch.Tensor:
    """Rotated-basis decoded fp16 matrix [in, out]."""
    ti, to, words = trellis.shape
    bits = words / 16
    tiles = trellis.reshape(-1, words)
    perm = _perm(trellis.device)
    out = torch.empty(tiles.shape[0], 256, dtype=torch.float16, device=trellis.device)
    for s in range(0, tiles.shape[0], 1 << 16):
        vals = _codebook(_states(tiles[s:s + (1 << 16)], bits), codebook)
        out[s:s + (1 << 16), perm] = vals
    return out.reshape(ti, to, 16, 16).permute(0, 2, 1, 3).reshape(ti * 16, to * 16)


_H = {}


def had128(device, dtype=torch.float32) -> torch.Tensor:
    key = (device, dtype)
    if key not in _H:
        h = torch.ones(1, 1, dtype=torch.float64)
        for _ in range(7):
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _H[key] = (h / 128 ** 0.5).to(device=device, dtype=dtype)
    return _H[key]


def dense(stored: dict, dtype=torch.float32, device="cuda") -> torch.Tensor:
    """W [in, out] with y = x @ W, computed in fp32 then cast to dtype."""
    codebook = "mul1" if "mul1" in stored else "mcg" if "mcg" in stored else "3inst"
    for k in ("mul1", "mcg"):
        if k in stored:
            want = MUL1 if k == "mul1" else MCG
            if (int(stored[k].reshape(-1)[0].item()) & 0xFFFFFFFF) != want:
                raise ValueError("unsupported codebook marker")
    w = inner(stored["trellis"].to(device), codebook).float()
    i, o = w.shape
    h = had128(device)
    w = (h @ w.reshape(i // 128, 128, o)).reshape(i, o)
    w = (w.reshape(i, o // 128, 128) @ h).reshape(i, o)
    w = w * stored["suh"].to(device).float()[:, None] * stored["svh"].to(device).float()[None, :]
    return w.to(dtype)
