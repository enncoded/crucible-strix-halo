"""CPU reference for the Qwen3.8-Flash-Next vision path (HF `qwen4_exp`, transformers 5.18.0).

It covers server preprocessing, the vision tower on dequantized EXL3 weights, and the
multimodal RoPE positions of a prompt. NumPy only, so the offline suite can import it.
`encode` evaluates in float64 with exact dense weights; it is an oracle, not a model of
device rounding. Sources: HF modeling_qwen4_exp.py (Qwen4ExpVisionModel, get_rope_index),
transformers vision_utils.py, Halogen 0.15.3 tools/serve_api.py (pixel limits and resize).
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import struct

import numpy as np

try:
    from . import exl3_ref
except ImportError:  # run as a script from ref/
    import exl3_ref

PATCH = 16
MERGE = 2
TEMPORAL = 2
UNIT = PATCH * MERGE  # both resized sides are multiples of 32
HIDDEN = 1152
HEADS = 16
HEAD_DIM = HIDDEN // HEADS  # 72
INTER = 4304  # HF intermediate size; EXL3 pads it to 4352
DEPTH = 27
OUT_HIDDEN = 2560
POS_SIDE = 48  # sqrt(num_position_embeddings = 2304)
VISION_THETA = 10000.0
LN_EPS = 1e-6

# Halogen 0.15.3 serve_api.py VIS_MIN_PIXELS / VIS_MAX_PIXELS. The checkpoint's
# preprocessor_config.json gives the same minimum (65536) and a 16,777,216 maximum.
MIN_PIXELS = 256 * 256
MAX_PIXELS = 2560 * 1440
MAX_ASPECT = 200

TEXT_THETA = 10000000.0
ROT_DIM = 64
MROPE_SECTION = (11, 11, 10)

_DTYPES = {'F16': np.float16, 'F32': np.float32, 'I16': np.int16, 'I32': np.int32, 'BF16': np.uint16}


# ---- checkpoint access -------------------------------------------------------

def read_safetensors(path: str, prefix: str = 'model.visual.') -> dict[str, np.ndarray]:
    """Minimal reader: BF16 becomes float32, other types keep their dtype."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    data = np.memmap(path, dtype=np.uint8, mode='r', offset=8 + n)
    out = {}
    for name, info in header.items():
        if name == '__metadata__' or not name.startswith(prefix):
            continue
        a, b = info['data_offsets']
        raw = np.frombuffer(data[a:b], dtype=_DTYPES[info['dtype']]).reshape(info['shape'])
        if info['dtype'] == 'BF16':
            raw = (raw.astype(np.uint32) << 16).view(np.float32)
        out[name[len(prefix):]] = np.array(raw)
    return out


def vision_file(ckpt: str) -> str:
    """The 4.05 pack keeps the tower in vision_k6.safetensors; 3.05 keeps it in a shard."""
    extra = os.path.join(ckpt, 'vision_k6.safetensors')
    if os.path.exists(extra):
        return extra
    index = json.load(open(os.path.join(ckpt, 'model.safetensors.index.json')))['weight_map']
    files = {v for k, v in index.items() if k.startswith('model.visual.')}
    if len(files) != 1:
        raise ValueError(f'expected the vision tower in one file, found {sorted(files)}')
    return os.path.join(ckpt, files.pop())


def exact_dense(stored: dict[str, np.ndarray]) -> np.ndarray:
    """W[in, out] = diag(suh) H inner H diag(svh) in float64, without fp16 reconstruction rounding."""
    inner = exl3_ref.reconstruct_inner(stored).astype(np.float64)
    h = exl3_ref._had_matrix().astype(np.float64)
    n_in, n_out = inner.shape
    left = (h @ inner.reshape(n_in // 128, 128, n_out)).reshape(n_in, n_out)
    left *= stored['suh'].astype(np.float64)[:, None]
    right = (left.reshape(n_in, n_out // 128, 128) @ h).reshape(n_in, n_out)
    return right * stored['svh'].astype(np.float64)[None, :]


LINEARS = ['attn.q_proj', 'attn.k_proj', 'attn.v_proj', 'attn.proj', 'mlp.linear_fc1', 'mlp.linear_fc2']


def load_weights(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Dense float64 weights in HF layout ([out, in] for linears, padding removed)."""
    w = {}

    def lin(key):
        stored = {s: tensors[f'{key}.{s}'] for s in ('trellis', 'suh', 'svh', 'mul1')}
        return exact_dense(stored).T, tensors[f'{key}.bias'].astype(np.float64)

    for i in range(DEPTH):
        p = f'blocks.{i}'
        for name in LINEARS:
            w[f'{p}.{name}.weight'], w[f'{p}.{name}.bias'] = lin(f'{p}.{name}')
        for n in ('norm1', 'norm2'):
            for s in ('weight', 'bias'):
                w[f'{p}.{n}.{s}'] = tensors[f'{p}.{n}.{s}'].astype(np.float64)
        pad_out = w[f'{p}.mlp.linear_fc1.weight'][INTER:]
        pad_in = w[f'{p}.mlp.linear_fc2.weight'][:, INTER:]
        if np.abs(pad_out).max() or np.abs(w[f'{p}.mlp.linear_fc1.bias'][INTER:]).max():
            raise ValueError(f'{p}: fc1 padding is not zero')
        w[f'{p}.mlp.linear_fc1.weight'] = w[f'{p}.mlp.linear_fc1.weight'][:INTER]
        w[f'{p}.mlp.linear_fc1.bias'] = w[f'{p}.mlp.linear_fc1.bias'][:INTER]
        w[f'{p}.mlp.linear_fc2.weight'] = w[f'{p}.mlp.linear_fc2.weight'][:, :INTER]
        w[f'{p}.mlp.pad_in_abs_max'] = np.array(np.abs(pad_in).max())
    for name in ('linear_fc1', 'linear_fc2'):
        w[f'merger.{name}.weight'], w[f'merger.{name}.bias'] = lin(f'merger.{name}')
    for s in ('weight', 'bias'):
        w[f'merger.norm.{s}'] = tensors[f'merger.norm.{s}'].astype(np.float64)
    w['patch_embed.proj.weight'] = tensors['patch_embed.proj.weight'].astype(np.float64).reshape(HIDDEN, -1)
    w['patch_embed.proj.bias'] = tensors['patch_embed.proj.bias'].astype(np.float64)
    w['pos_embed.weight'] = tensors['pos_embed.weight'].astype(np.float64)
    return w


# ---- preprocessing -----------------------------------------------------------

def smart_resize(height: int, width: int, min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """transformers qwen2_vl smart_resize with factor 32 (the same arithmetic as Halogen)."""
    if min(height, width) <= 0:
        raise ValueError('image has a zero dimension')
    if max(height, width) / min(height, width) > MAX_ASPECT:
        raise ValueError(f'image aspect ratio must be under {MAX_ASPECT}:1')
    h_bar = max(UNIT, round(height / UNIT) * UNIT)
    w_bar = max(UNIT, round(width / UNIT) * UNIT)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(UNIT, math.floor(height / beta / UNIT) * UNIT)
        w_bar = max(UNIT, math.floor(width / beta / UNIT) * UNIT)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / UNIT) * UNIT
        w_bar = math.ceil(width * beta / UNIT) * UNIT
    return h_bar, w_bar


def resize_rgb(image, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> np.ndarray:
    """PIL image -> resized uint8 [H, W, 3] (BICUBIC on 8-bit data, as Halogen and ExLlamaV3)."""
    from PIL import Image
    image = image.convert('RGB')
    w0, h0 = image.size
    h, w = smart_resize(h0, w0, min_pixels, max_pixels)
    if (w, h) != (w0, h0):
        image = image.resize((w, h), Image.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def patchify(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """uint8 [H, W, 3] -> float32 patches [gh*gw, 3*2*16*16] in merge-block order, grid (1, gh, gw)."""
    h, w, c = rgb.shape
    if c != 3 or h % UNIT or w % UNIT:
        raise ValueError('rgb must be [H, W, 3] with sides divisible by 32')
    x = (rgb.astype(np.float32) * np.float32(1 / 255) - np.float32(0.5)) / np.float32(0.5)
    x = np.broadcast_to(x.transpose(2, 0, 1)[None], (TEMPORAL, 3, h, w))  # one image = two equal frames
    gh, gw = h // PATCH, w // PATCH
    x = x.reshape(1, TEMPORAL, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return np.ascontiguousarray(x.reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH)), (1, gh, gw)


def image_digest(rgb: np.ndarray) -> bytes:
    """Cache identity of one resized image: sha256 over its shape and pixels."""
    h, w, _ = rgb.shape
    return hashlib.sha256(struct.pack('<II', h, w) + np.ascontiguousarray(rgb).tobytes()).digest()


def decode_image_bytes(raw: bytes):
    from PIL import Image
    image = Image.open(io.BytesIO(raw))
    image.load()
    return image


# ---- vision tower ------------------------------------------------------------

def patch_rows_cols(gh: int, gw: int) -> tuple[np.ndarray, np.ndarray]:
    """Patch (row, col) for each patch in merge-block order."""
    i = np.arange(gh * gw)
    in_col, in_row = i % MERGE, (i // MERGE) % MERGE
    block_col = (i // (MERGE * MERGE)) % (gw // MERGE)
    block_row = i // (MERGE * MERGE * (gw // MERGE))
    return block_row * MERGE + in_row, block_col * MERGE + in_col


def pos_embed_interp(pos_table: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """Bilinear, align_corners=True resample of the 48x48 table to (gh, gw), merge-block order."""
    rows, cols = patch_rows_cols(gh, gw)

    def taps(index, size):
        src = index.astype(np.float32) * np.float32(POS_SIDE - 1) / np.float32(max(size - 1, 1))
        lo = np.floor(src)
        t = np.stack([lo, lo + 1], 1).astype(np.int64)
        wt = np.clip(1 - np.abs(src[:, None] - t), 0, None)
        return np.clip(t, 0, POS_SIDE - 1), wt.astype(np.float64)

    ht, hw = taps(rows, gh)
    wt, ww = taps(cols, gw)
    idx = (ht[:, :, None] * POS_SIDE + wt[:, None, :]).reshape(-1, 4)
    wgt = (hw[:, :, None] * ww[:, None, :]).reshape(-1, 4)
    return (pos_table[idx] * wgt[:, :, None]).sum(1)


def vision_rope(gh: int, gw: int) -> tuple[np.ndarray, np.ndarray]:
    """cos, sin [N, 72]: channels 0..17 row, 18..35 col, repeated for the rotate-half pair."""
    rows, cols = patch_rows_cols(gh, gw)
    inv = 1.0 / (VISION_THETA ** (np.arange(0, HEAD_DIM // 2, 2, dtype=np.float64) / (HEAD_DIM // 2)))
    f = np.concatenate([rows[:, None] * inv, cols[:, None] * inv], 1)
    f = np.concatenate([f, f], 1)
    return np.cos(f), np.sin(f)


def layer_norm(x, w, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + LN_EPS) * w + b


def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def gelu_erf(x):
    try:
        from scipy.special import erf
    except ImportError:  # NumPy only: exact but slow
        from math import erf
        erf = np.vectorize(erf)
    return 0.5 * x * (1 + erf(x / math.sqrt(2)))


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], -1)


def attention(x, w, p, cos, sin):
    n = x.shape[0]
    q = (x @ w[f'{p}.attn.q_proj.weight'].T + w[f'{p}.attn.q_proj.bias']).reshape(n, HEADS, HEAD_DIM)
    k = (x @ w[f'{p}.attn.k_proj.weight'].T + w[f'{p}.attn.k_proj.bias']).reshape(n, HEADS, HEAD_DIM)
    v = (x @ w[f'{p}.attn.v_proj.weight'].T + w[f'{p}.attn.v_proj.bias']).reshape(n, HEADS, HEAD_DIM)
    q = q * cos[:, None] + rotate_half(q) * sin[:, None]
    k = k * cos[:, None] + rotate_half(k) * sin[:, None]
    out = np.empty_like(q)
    for h in range(HEADS):
        s = (q[:, h] @ k[:, h].T) / math.sqrt(HEAD_DIM)
        s = np.exp(s - s.max(1, keepdims=True))
        out[:, h] = (s / s.sum(1, keepdims=True)) @ v[:, h]
    return out.reshape(n, HIDDEN) @ w[f'{p}.attn.proj.weight'].T + w[f'{p}.attn.proj.bias']


def encode(w: dict[str, np.ndarray], patches: np.ndarray, grid: tuple[int, int, int],
           blocks: int = DEPTH, trace: dict | None = None) -> np.ndarray:
    """One image's patches -> merged embeddings [gh*gw/4, 2560], float64."""
    _, gh, gw = grid
    x = patches.astype(np.float64) @ w['patch_embed.proj.weight'].T + w['patch_embed.proj.bias']
    x = x + pos_embed_interp(w['pos_embed.weight'], gh, gw)
    if trace is not None:
        trace['embed'] = x.copy()
    cos, sin = vision_rope(gh, gw)
    for i in range(blocks):
        p = f'blocks.{i}'
        x = x + attention(layer_norm(x, w[f'{p}.norm1.weight'], w[f'{p}.norm1.bias']), w, p, cos, sin)
        hmid = layer_norm(x, w[f'{p}.norm2.weight'], w[f'{p}.norm2.bias'])
        hmid = gelu_tanh(hmid @ w[f'{p}.mlp.linear_fc1.weight'].T + w[f'{p}.mlp.linear_fc1.bias'])
        x = x + hmid @ w[f'{p}.mlp.linear_fc2.weight'].T + w[f'{p}.mlp.linear_fc2.bias']
        if trace is not None:
            trace[f'block{i}'] = x.copy()
    x = layer_norm(x, w['merger.norm.weight'], w['merger.norm.bias']).reshape(-1, HIDDEN * MERGE * MERGE)
    x = gelu_erf(x @ w['merger.linear_fc1.weight'].T + w['merger.linear_fc1.bias'])
    return x @ w['merger.linear_fc2.weight'].T + w['merger.linear_fc2.bias']


# ---- prompt layout and multimodal RoPE --------------------------------------

def image_tokens(grid: tuple[int, int, int]) -> int:
    _, gh, gw = grid
    return (gh // MERGE) * (gw // MERGE)


def mrope_positions(ids: list[int], grids: list[tuple[int, int, int]], image_token_id: int) -> np.ndarray:
    """[3, L] (T, H, W) positions, as HF get_rope_index for images.

    Each maximal run of image tokens is one image, in order. An image that starts at position s
    gives merged cell (r, c) the triple (s, s + r, s + c); the next token starts at
    s + max(gh, gw) / 2. Text tokens get three equal values.
    """
    out = np.zeros((3, len(ids)), dtype=np.int64)
    pos, i, k = 0, 0, 0
    while i < len(ids):
        if ids[i] != image_token_id:
            out[:, i] = pos
            pos += 1
            i += 1
            continue
        if k >= len(grids):
            raise ValueError('more image token runs than images')
        _, gh, gw = grids[k]
        hh, ww = gh // MERGE, gw // MERGE
        n = hh * ww
        if ids[i:i + n] != [image_token_id] * n:
            raise ValueError(f'image {k} needs {n} consecutive image tokens')
        r, c = np.divmod(np.arange(n), ww)
        out[0, i:i + n] = pos
        out[1, i:i + n] = pos + r
        out[2, i:i + n] = pos + c
        pos += max(hh, ww)
        i += n
        k += 1
    if k != len(grids):
        raise ValueError('fewer image token runs than images')
    return out


def mrope_axis(j: int) -> int:
    """Interleaved sections [11, 11, 10]: which axis (0 T, 1 H, 2 W) drives frequency j."""
    if j % 3 == 1 and j < 3 * MROPE_SECTION[1]:
        return 1
    if j % 3 == 2 and j < 3 * MROPE_SECTION[2]:
        return 2
    return 0


def text_rope_angles(pos3: np.ndarray) -> np.ndarray:
    """[L, 32] float64 angles of the partial text RoPE for (T, H, W) positions [3, L]."""
    inv = TEXT_THETA ** (-2.0 * np.arange(ROT_DIM // 2) / ROT_DIM)
    axes = np.array([mrope_axis(j) for j in range(ROT_DIM // 2)])
    return pos3[axes].T * inv[None, :]
