"""Offline checks of the vision reference that need no checkpoint and no PyTorch."""
import numpy as np
import pytest

import vision_ref as v

IMG = 248056


@pytest.mark.parametrize('h,w,expected', [
    (1080, 1920, (1088, 1920)),
    (800, 1280, (800, 1280)),
    (100, 100, (256, 256)),            # raised to the 256x256 minimum
    (2160, 3840, (1440, 2560)),        # lowered to the 2560x1440 maximum
    (31, 6000, (32, 6016)),
    (1, 1, (256, 256)),
])
def test_smart_resize(h, w, expected):
    assert v.smart_resize(h, w) == expected


@pytest.mark.parametrize('h,w', [(0, 5), (5, 0), (1, 201)])
def test_smart_resize_rejects(h, w):
    with pytest.raises(ValueError):
        v.smart_resize(h, w)


def test_smart_resize_limits_hold():
    rng = np.random.default_rng(3)
    for _ in range(2000):
        h, w = (int(x) for x in rng.integers(1, 5000, 2))
        if max(h, w) / min(h, w) > 200:
            continue
        rh, rw = v.smart_resize(h, w)
        assert rh % 32 == 0 and rw % 32 == 0
        assert rh * rw <= v.MAX_PIXELS


def test_patchify_order():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    patches, grid = v.patchify(rgb)
    assert grid == (1, 4, 6)
    assert patches.shape == (24, 1536)
    norm = (rgb.astype(np.float32) / 255 - 0.5) / 0.5
    rows, cols = v.patch_rows_cols(4, 6)
    for i in range(24):
        r, c = rows[i], cols[i]
        block = norm[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16].transpose(2, 0, 1)  # [3,16,16]
        expect = np.stack([block, block], 1).reshape(-1)  # channel, then temporal copy, then pixels
        np.testing.assert_allclose(patches[i], expect, atol=1e-6)
    # The first four patches form one 2x2 merge cell.
    assert list(zip(rows[:4], cols[:4])) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_mrope_text_only_is_index():
    ids = list(range(100, 140))
    pos = v.mrope_positions(ids, [], IMG)
    assert (pos == np.arange(40)[None]).all()


def test_mrope_one_image():
    grid = (1, 6, 10)  # 3x5 merged cells
    ids = [1, 2, 248053] + [IMG] * 15 + [248054, 7, 8]
    pos = v.mrope_positions(ids, [grid], IMG)
    s = 3
    assert (pos[:, :3] == np.arange(3)).all()
    r, c = np.divmod(np.arange(15), 5)
    assert (pos[0, 3:18] == s).all()
    assert (pos[1, 3:18] == s + r).all()
    assert (pos[2, 3:18] == s + c).all()
    # The text after the image continues at s + max(3, 5).
    assert (pos[:, 18:] == np.array([s + 5, s + 6, s + 7])).all()


def test_mrope_two_adjacent_images():
    ids = [IMG] * 4 + [IMG] * 2 + [5]
    pos = v.mrope_positions(ids, [(1, 4, 4), (1, 2, 4)], IMG)
    assert pos[:, 4].tolist() == [2, 2, 2]
    assert pos[:, 6].tolist() == [4, 4, 4]


def test_mrope_count_mismatch():
    with pytest.raises(ValueError):
        v.mrope_positions([IMG] * 3, [(1, 4, 4)], IMG)
    with pytest.raises(ValueError):
        v.mrope_positions([1, 2], [(1, 4, 4)], IMG)


def test_mrope_axes():
    axes = [v.mrope_axis(j) for j in range(32)]
    assert axes.count(1) == 11 and axes.count(2) == 10 and axes.count(0) == 11
    assert axes[:6] == [0, 1, 2, 0, 1, 2]
    assert axes[29:] == [2, 0, 1]


def test_text_angles_equal_positions_match_1d():
    pos = np.arange(50)
    angles = v.text_rope_angles(np.stack([pos, pos, pos]))
    inv = 10000000.0 ** (-2.0 * np.arange(32) / 64)
    np.testing.assert_array_equal(angles, pos[:, None] * inv[None])


def test_pos_embed_interp_identity_at_48():
    table = np.random.default_rng(1).normal(size=(48 * 48, 8))
    out = v.pos_embed_interp(table, 48, 48)
    rows, cols = v.patch_rows_cols(48, 48)
    np.testing.assert_allclose(out, table[rows * 48 + cols], atol=1e-12)


def test_image_digest_depends_on_pixels_and_shape():
    a = np.zeros((32, 64, 3), np.uint8)
    b = a.copy()
    b[5, 5, 1] = 1
    assert v.image_digest(a) != v.image_digest(b)
    assert v.image_digest(a) != v.image_digest(np.zeros((64, 32, 3), np.uint8))
