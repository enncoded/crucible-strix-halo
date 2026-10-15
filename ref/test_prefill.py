"""Algebra and visibility gates. GPU harnesses independently check the HIP code."""
import numpy as np
import pytest
from prefill_ref import chunk_gdn, serial_gdn, dispatch, selected_tokens


@pytest.mark.parametrize("rows", [1, 3, 64, 65, 130])
def test_chunk_gdn(rows):
    rng = np.random.default_rng(12)
    q, k, v = [rng.normal(size=(rows, 8)) for _ in range(3)]
    q /= np.linalg.norm(q, axis=-1, keepdims=True) * np.sqrt(8)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    g = -rng.uniform(0, 2, rows)
    beta = rng.uniform(0, 1, rows)
    initial = rng.normal(size=(8, 8))
    ys, ss = serial_gdn(q, k, v, g, beta, initial)
    yc, sc = chunk_gdn(q, k, v, g, beta, initial)
    np.testing.assert_allclose(yc, ys, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(sc, ss, rtol=2e-13, atol=2e-14)


def test_decay_underflow():
    q = k = v = np.ones((64, 2)) / np.sqrt(2)
    yc, sc = chunk_gdn(q, k, v, np.full(64, -100.), np.ones(64), np.ones((2, 2)))
    ys, ss = serial_gdn(q, k, v, np.full(64, -100.), np.ones(64), np.ones((2, 2)))
    assert np.isfinite(yc).all()
    np.testing.assert_allclose(yc, ys)
    np.testing.assert_allclose(sc, ss)


def test_dispatch_stable():
    selections = np.array([[2, 1, 0], [0, 2, 1], [1, 0, 2]])
    counts, offsets, pairs = dispatch(selections, 4)
    assert counts.tolist() == [3, 3, 3, 0]
    assert offsets.tolist() == [0, 3, 6, 9, 9]
    assert pairs.tolist() == [2, 3, 7, 1, 5, 6, 0, 4, 8]
    with pytest.raises(ValueError):
        dispatch(np.array([[4]]), 4)


@pytest.mark.parametrize("n", [2047, 2048, 2049, 2051, 2052, 2055, 2060])
def test_qsa_visibility(n):
    scores = np.zeros(n // 4)
    if n // 4 > 512:
        scores[512] = 1
    tokens = selected_tokens(scores, n)
    assert len(tokens) == min(n // 4, 512) * 4 + n % 4
    assert np.all(tokens < n)
    assert np.all(np.diff(tokens) > 0)
    if n // 4 > 512:
        assert 512 * 4 in tokens
        assert 511 * 4 not in tokens
    assert tokens[-(n % 4):].tolist() == list(range(n - n % 4, n)) if n % 4 else True
