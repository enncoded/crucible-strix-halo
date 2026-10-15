"""NumPy reference for EXL3 linear tensors, source commit d3739fd.

Codebook values and decoder states are bit-exact. reconstruct models the Python
get_weight_tensor path, not the different rounding of fused CUDA reconstruction.
matvec follows the default FP16 activation transform / decoded matmul / output
transform order. Its middle dot uses FP32 BLAS, not NVIDIA MMA accumulation.
"""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np

HAD_SIZE = 128
HAD_SCALE = np.float32(0.088388347648)
MUL1 = 0x83DCD12D
MCG = 0xCBAC1FED


def _bitrate(bits: float) -> tuple[int, bool]:
    bits = float(bits)
    if not np.isfinite(bits) or bits * 2 != int(bits * 2):
        raise ValueError('EXL3 bitrate must be integer or half-integer')
    whole = int(bits)
    half = bits != whole
    if not (1 <= whole <= (3 if half else 8)):
        raise ValueError('Supported bitrates: 1..8 and 1.5, 2.5, 3.5')
    return whole, half


def tensor_core_perm() -> np.ndarray:
    """Position i in decoder order maps to row-major tile element perm[i]."""
    lane = np.arange(32)
    rows = 2 * (lane % 4)
    cols = lane // 4
    return np.stack([rows * 16 + cols, (rows + 1) * 16 + cols,
                     (rows + 8) * 16 + cols, (rows + 9) * 16 + cols,
                     rows * 16 + cols + 8, (rows + 1) * 16 + cols + 8,
                     (rows + 8) * 16 + cols + 8, (rows + 9) * 16 + cols + 8], axis=-1).reshape(256)


def decode_tile_symbols(packed: np.ndarray, bits: float | None = None) -> np.ndarray:
    """Return 256 uint16 decoder states per tile, in tensor-core order.

    packed[..., 16*K] contains little-endian int16 storage words. Despite the
    name, the result is the full sliding 16-bit state, not just its K new bits.
    uint32 words are consumed MSB first; state i ends at the end of symbol i.
    The 16-bit window wraps around within the same tile.
    """
    packed = np.asarray(packed)
    if packed.ndim < 1 or packed.dtype.kind not in 'iu' or packed.dtype.itemsize != 2:
        raise ValueError('packed must contain 16-bit integers')
    bits = packed.shape[-1] / 16 if bits is None else bits
    whole, half = _bitrate(bits)
    if packed.shape[-1] != 16 * float(bits):
        raise ValueError('Packed word count does not match bitrate')
    words16 = packed.astype('<u2', copy=False)
    words = words16[..., 0::2].astype(np.uint32) | (words16[..., 1::2].astype(np.uint32) << 16)
    widths = whole + (np.arange(256) % 2 if half else np.zeros(256, dtype=np.int64))
    ends = np.cumsum(widths) + int(256 * float(bits))
    starts = ends - 16
    first = starts // 32
    last = (ends - 1) // 32
    shift = (last + 1) * 32 - ends
    a = words[..., first % words.shape[-1]].astype(np.uint64)
    b = words[..., last % words.shape[-1]].astype(np.uint64)
    return (((a << 32 | b) >> shift.astype(np.uint64)) & 65535).astype(np.uint16)


def decode_codebook(states: np.ndarray, codebook: str = 'mul1') -> np.ndarray:
    """Decode states to exact fp16 codebook values, including half FMA rounding."""
    states = np.asarray(states, dtype=np.uint16).astype(np.uint64)
    if codebook == 'mul1':
        product = (states * MUL1) & 0xffffffff
        byte_sum = sum((product >> shift) & 255 for shift in [0, 8, 16, 24])
        h = (np.uint16(0x6400) + byte_sum.astype(np.uint16)).view(np.float16)
        inv = np.array(0x1eee, dtype=np.uint16).view(np.float16).astype(np.float64)
        bias = np.array(0xc931, dtype=np.uint16).view(np.float16).astype(np.float64)
        # Binary64 represents these fp16 operands' product + sum exactly.
        return (h.astype(np.float64) * inv + bias).astype(np.float16)
    if codebook not in ['mcg', '3inst']:
        raise ValueError('Unknown codebook')
    product = (states * (MCG if codebook == 'mcg' else 89226354) +
               (0 if codebook == 'mcg' else 64248484)) & 0xffffffff
    value = (product & 0x8fff8fff) ^ 0x3b603b60
    low = (value & 65535).astype(np.uint16).view(np.float16)
    high = (value >> 16).astype(np.uint16).view(np.float16)
    return (low.astype(np.float32) + high.astype(np.float32)).astype(np.float16)


def _stored(stored: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    trellis = np.asarray(stored['trellis'])
    suh, svh = np.asarray(stored['suh']), np.asarray(stored['svh'])
    if trellis.ndim != 3 or trellis.shape[:2] != (suh.size // 16, svh.size // 16):
        raise ValueError('Trellis shape disagrees with scales')
    if suh.ndim != 1 or svh.ndim != 1 or suh.size % 128 or svh.size % 128:
        raise ValueError('Linear dimensions must be multiples of 128')
    if suh.dtype != np.float16 or svh.dtype != np.float16:
        raise ValueError('suh and svh must be fp16')
    codebook = 'mul1' if 'mul1' in stored else 'mcg' if 'mcg' in stored else '3inst'
    if 'mul1' in stored and 'mcg' in stored:
        raise ValueError('Ambiguous codebook markers')
    if codebook in ['mul1', 'mcg']:
        marker = np.asarray(stored[codebook])
        if marker.size != 1 or (int(marker.item()) & 0xffffffff) != (MUL1 if codebook == 'mul1' else MCG):
            raise ValueError('Unsupported codebook multiplier marker')
    if trellis.shape[-1] % 16 and codebook != 'mul1':
        raise ValueError('Half-integer bitrates require mul1')
    return trellis, suh, svh, codebook


def reconstruct_inner(stored_tensors: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return the rotated-basis decoded fp16 matrix [input, output]."""
    trellis, suh, svh, codebook = _stored(stored_tensors)
    tiles = trellis.reshape(-1, trellis.shape[-1])
    perm = tensor_core_perm()
    decoded = np.empty((tiles.shape[0], 256), dtype=np.float16)
    for first in range(0, tiles.shape[0], 1024):
        states = decode_tile_symbols(tiles[first:first + 1024])
        decoded[first:first + 1024, perm] = decode_codebook(states, codebook)
    return decoded.reshape(trellis.shape[0], trellis.shape[1], 16, 16).transpose(0, 2, 1, 3).reshape(suh.size, svh.size)


def _had_matrix() -> np.ndarray:
    h = np.ones((1, 1), dtype=np.float32)
    for _ in range(7):
        h = np.block([[h, h], [h, -h]])
    return h * HAD_SCALE


def reconstruct(stored_tensors: Mapping[str, np.ndarray]) -> np.ndarray:
    """get_weight_tensor rounding: fp32 Had matmul, fp16 after each transform/scale."""
    _, suh, svh, _ = _stored(stored_tensors)
    inner = reconstruct_inner(stored_tensors)
    h = _had_matrix()
    left = (h @ inner.astype(np.float32).reshape(-1, 128, svh.size)).reshape(inner.shape).astype(np.float16)
    left_scaled = (left.astype(np.float32) * suh.astype(np.float32)[:, None]).astype(np.float16)
    right = (left_scaled.astype(np.float32).reshape(suh.size, -1, 128) @ h).reshape(inner.shape).astype(np.float16)
    return (right.astype(np.float32) * svh.astype(np.float32)[None, :]).astype(np.float16).astype(np.float32)


def hadamard_kernel(x: np.ndarray, pre_scale: np.ndarray | None = None,
                    post_scale: np.ndarray | None = None) -> np.ndarray:
    """had_hf / had_ff natural-order 128 butterfly and its conversion points."""
    x = np.asarray(x)
    if x.dtype not in [np.dtype('float16'), np.dtype('float32')] or x.shape[-1] % 128:
        raise ValueError('Hadamard input must be fp16/fp32 with 128-divisible width')
    dtype = x.dtype
    scaled = x if pre_scale is None else (x.astype(np.float32) * np.asarray(pre_scale, dtype=np.float32)).astype(dtype)
    v = scaled.astype(np.float32).reshape(*x.shape[:-1], -1, 128)
    for stride in [1, 2, 4, 8, 16, 32, 64]:
        pair = v.reshape(*v.shape[:-1], -1, 2, stride)
        low, high = pair[..., 0, :], pair[..., 1, :]
        v = np.stack([low + high, low - high], axis=-2).reshape(v.shape)
    result = (v * HAD_SCALE).reshape(x.shape).astype(dtype)
    if post_scale is not None:
        result = (result.astype(np.float32) * np.asarray(post_scale, dtype=np.float32)).astype(dtype)
    return result


def matvec(x: np.ndarray, stored_tensors: Mapping[str, np.ndarray],
           out_dtype: np.dtype = np.float16) -> np.ndarray:
    """GEMV stage order; FP32 BLAS dot is a portable, not MMA-bit-exact, oracle.

    out_dtype is activation metadata, not encoded in the EXL3 weight tensors.
    Routed expert down projections use float32 output in ExLlamaV3.
    """
    _, suh, svh, _ = _stored(stored_tensors)
    x = np.asarray(x, dtype=np.float16)
    if x.shape != (suh.size,):
        raise ValueError('x must be one input vector')
    xh = hadamard_kernel(x, pre_scale=suh)
    dtype = np.dtype(out_dtype)
    if dtype not in [np.dtype('float16'), np.dtype('float32')]:
        raise ValueError('out_dtype must be fp16 or fp32')
    rotated = (xh.astype(np.float32) @ reconstruct_inner(stored_tensors).astype(np.float32)).astype(dtype)
    output = hadamard_kernel(rotated, post_scale=svh)
    if 'bias' in stored_tensors:
        output = (output.astype(np.float32) + np.asarray(stored_tensors['bias'], dtype=np.float32)).astype(dtype)
    return output
