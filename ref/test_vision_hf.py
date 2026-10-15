"""Checks vision_ref against transformers 5.18.0 qwen4_exp on dequantized EXL3 weights.

Needs PyTorch, transformers==5.18.0, Pillow, and CRUCIBLE_VISION_DIR: a directory with the
checkpoint config.json and either the checkpoint's vision shard or vision_k6.safetensors
(CRUCIBLE_VISION_FILE overrides the file). transformers 5.8.0, the config's version, has no qwen4_exp.
"""
import os
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest

import vision_ref as v

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
PIL = pytest.importorskip('PIL')
from PIL import Image, ImageDraw  # noqa: E402

VDIR = os.environ.get('CRUCIBLE_VISION_DIR')
pytestmark = pytest.mark.skipif(not VDIR, reason='set CRUCIBLE_VISION_DIR')


def synthetic_image(w, h, seed):
    """Deterministic test picture with gradients, shapes, and text."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([xx * 255 // max(w - 1, 1), yy * 255 // max(h - 1, 1), (xx + yy) % 256], -1)
    im = Image.fromarray(base.astype(np.uint8))
    d = ImageDraw.Draw(im)
    for _ in range(12):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = x0 + int(rng.integers(5, w // 2 + 6)), y0 + int(rng.integers(5, h // 2 + 6))
        d.rectangle([x0, y0, x1, y1], fill=tuple(int(c) for c in rng.integers(0, 256, 3)))
    d.text((5, 5), 'def f(x): return x + 1', fill=(0, 0, 0))
    return im


@lru_cache(maxsize=1)
def weights():
    path = os.environ.get('CRUCIBLE_VISION_FILE') or v.vision_file(VDIR)
    return v.load_weights(v.read_safetensors(path))


@lru_cache(maxsize=1)
def hf_config():
    return transformers.AutoConfig.from_pretrained(VDIR)


@lru_cache(maxsize=1)
def hf_vision():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel
    cfg = hf_config().vision_config
    cfg._attn_implementation = 'eager'
    model = Qwen4ExpVisionModel(cfg).to(torch.float64).eval()
    w = weights()
    sd = {}
    for k, val in w.items():
        if k.endswith('pad_in_abs_max'):
            continue
        sd[k] = torch.from_numpy(np.ascontiguousarray(val))
    for i in range(v.DEPTH):
        p = f'blocks.{i}.attn'
        sd[f'{p}.qkv.weight'] = torch.cat([sd.pop(f'{p}.{n}_proj.weight') for n in 'qkv'])
        sd[f'{p}.qkv.bias'] = torch.cat([sd.pop(f'{p}.{n}_proj.bias') for n in 'qkv'])
    sd['patch_embed.proj.weight'] = sd['patch_embed.proj.weight'].reshape(v.HIDDEN, 3, 2, 16, 16)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and not [m for m in missing if 'inv_freq' not in m], (missing, unexpected)
    return model


def cosine(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def test_fc1_padding_outputs_are_zero():
    # svh and bias are zero on the 48 padded fc1 outputs, so GELU feeds exact zeros into the
    # padded fc2 input rows. Those fc2 rows are not zero, so the zeros are what makes them inert.
    t = v.read_safetensors(os.environ.get('CRUCIBLE_VISION_FILE') or v.vision_file(VDIR))
    for i in range(v.DEPTH):
        assert not t[f'blocks.{i}.mlp.linear_fc1.svh'][v.INTER:].any()
        assert not t[f'blocks.{i}.mlp.linear_fc1.bias'][v.INTER:].any()


@pytest.mark.parametrize('w,h', [(256, 256), (500, 300), (1000, 120)])
def test_preprocess_matches_hf_pil_processor(w, h):
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    proc = Qwen2VLImageProcessorPil(patch_size=16, temporal_patch_size=2, merge_size=2,
                                    image_mean=[0.5] * 3, image_std=[0.5] * 3,
                                    size={'shortest_edge': v.MIN_PIXELS, 'longest_edge': v.MAX_PIXELS})
    im = synthetic_image(w, h, 7)
    got = proc(images=[im], return_tensors='np')
    patches, grid = v.patchify(v.resize_rgb(im))
    assert tuple(got['image_grid_thw'][0]) == grid
    np.testing.assert_allclose(got['pixel_values'], patches, atol=2e-6)


@pytest.mark.parametrize('w,h,seed', [(256, 256, 1), (384, 192, 2)])
def test_encoder_matches_hf(w, h, seed):
    patches, grid = v.patchify(v.resize_rgb(synthetic_image(w, h, seed)))
    trace = {}
    ours = v.encode(weights(), patches, grid, trace=trace)
    with torch.no_grad():
        out = hf_vision()(torch.from_numpy(patches.astype(np.float64)), grid_thw=torch.tensor([grid]),
                          output_hidden_states=True)
    hf = out.pooler_output.numpy()
    assert ours.shape == hf.shape == (grid[1] * grid[2] // 4, v.OUT_HIDDEN)
    err = np.abs(ours - hf).max() / np.abs(hf).max()
    print(f'{w}x{h}: max rel err {err:.3e}, cos {cosine(ours, hf):.12f}')
    assert err < 1e-4 and cosine(ours, hf) > 0.999999


def test_rope_index_matches_hf():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel
    cfg = hf_config()
    me = SimpleNamespace(config=cfg)
    me.get_vision_position_ids = lambda *a, **k: Qwen4ExpModel.get_vision_position_ids(me, *a, **k)
    grids = [(1, 6, 10), (1, 8, 4), (1, 16, 16)]
    ids, ttype = [], []

    def text(n):
        ids.extend(range(1000, 1000 + n))
        ttype.extend([0] * n)

    text(5)
    for g in grids:
        ids.append(cfg.vision_start_token_id); ttype.append(0)
        n = v.image_tokens(g)
        ids.extend([cfg.image_token_id] * n); ttype.extend([1] * n)
        ids.append(cfg.vision_end_token_id); ttype.append(0)
        text(3)
    hf, delta = Qwen4ExpModel.get_rope_index(me, torch.tensor([ids]), torch.tensor([ttype]),
                                             image_grid_thw=torch.tensor(grids))
    ours = v.mrope_positions(ids, grids, cfg.image_token_id)
    np.testing.assert_array_equal(hf[:, 0].numpy(), ours)
    assert int(delta[0, 0]) == int(ours.max()) + 1 - len(ids)


def test_text_rope_matches_hf():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRotaryEmbedding
    rot = Qwen4ExpTextRotaryEmbedding(hf_config().text_config)
    ids = [1, 2, hf_config().image_token_id] * 1 + [hf_config().image_token_id] * 23 + [9] * 6
    pos = v.mrope_positions(ids, [(1, 8, 12)], hf_config().image_token_id)
    cos, sin = rot(torch.zeros(1, dtype=torch.float32), torch.from_numpy(pos[:, None, :]))
    ang = v.text_rope_angles(pos)
    np.testing.assert_allclose(cos[0, :, :32].numpy(), np.cos(ang), atol=2e-6)
    np.testing.assert_allclose(sin[0, :, 32:].numpy(), np.sin(ang), atol=2e-6)
