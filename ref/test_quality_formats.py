"""Offline format gates for the lane window and both PLE table layouts."""
from unittest.mock import patch

import numpy as np
import pytest
import torch

from exl3_ref import decode_codebook, decode_tile_symbols
from oracle import ple_embed


@pytest.mark.parametrize("K", [2, 3, 4, 5, 6])
def test_lane_window(K):
    rng = np.random.default_rng(405)
    words = rng.integers(0, 2**32, size=(32, 8 * K), dtype=np.uint32)
    expected = decode_tile_symbols(words.view(np.uint16))
    actual = np.empty_like(expected)

    def align(a, b, shift):
        assert 0 <= shift < 32
        return (((int(a) << 32) | int(b)) >> shift) & 0xffffffff

    for t in range(32):
        start = (8 * t + 1) * K - 16
        dw = (start - 1) // 32
        shift = start - dw * 32
        assert 1 <= shift <= 32
        assert shift + 7 * K + 16 <= 96
        for tile, row in enumerate(words):
            w0, w1, w2 = [row[(dw + i) % len(row)] for i in range(3)]
            a0 = align(w0, w1, 32 - shift)
            a1 = align(w1, w2, 32 - shift)
            for j in range(8):
                end = j * K + 16
                state = a0 >> (32 - end) if end <= 32 else align(a0, a1, 64 - end)
                actual[tile, t * 8 + j] = state & 65535
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("K", [5, 6])
@pytest.mark.parametrize("single", [False, True])
def test_ple_oracle_layout(K, single):
    rng = np.random.default_rng(405)
    symbols = rng.integers(0, 1 << K, size=(16, 160), dtype=np.uint16)
    scales = np.linspace(0.01, 1.0, 16).astype(np.float16)
    bias = rng.normal(size=(16, 160)).astype(np.float16)
    bits = ((symbols[:, :, None] >> np.arange(K)) & 1).reshape(16, -1, 16)
    words = np.sum(bits.astype(np.uint32) << np.arange(16), axis=-1).astype(np.uint16)
    packed = np.column_stack([scales.view(np.uint16), words]).view(np.int16)
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
    if single:
        tensors = {prefix + "trellis": torch.from_numpy(packed)}
    else:
        tensors = {prefix + f"shard_{i}.trellis": torch.from_numpy(packed[i * 8:(i + 1) * 8])
                   for i in range(2)}
    tensors[prefix + "head_bias"] = torch.from_numpy(bias)

    class FakeCheckpoint:
        map = tensors

        def t(self, name):
            return tensors[name]

        def f64(self, name):
            return tensors[name].numpy().astype(np.float64)

    states = np.zeros((16, 160), dtype=np.uint16)
    for i in range(160):
        for older in range((16 + K - 1) // K):
            states[:, i] |= (symbols[:, (i - older) % 160].astype(np.uint32)
                             << (older * K)).astype(np.uint16)
    expected = decode_codebook(states).astype(np.float64) * scales[:, None].astype(np.float64) + bias
    with patch("oracle.ple_rows", return_value=[list(range(16))]):
        actual = ple_embed(FakeCheckpoint(), [0])
    np.testing.assert_array_equal(actual.reshape(16, 160), expected)
