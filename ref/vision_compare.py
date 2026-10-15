"""Prepare encoder inputs with float64 references, and compare device bf16 output against them.

  python ref/vision_compare.py prepare CKPT_OR_VISION_FILE OUTDIR
  python ref/vision_compare.py compare OUTDIR
"""
import json
import os
import sys

import numpy as np

import vision_ref as v

# (name, width, height, reference?) The reference is float64 and quadratic in patches.
SIZES = [("square256", 256, 256, True), ("wide384", 384, 192, True), ("screen1280", 1280, 800, True),
         ("chunked1600", 1600, 900, True), ("screen1080p", 1920, 1080, False),
         ("screen1440p", 2560, 1440, False)]


def picture(width, height, seed):
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack([xx * 255 // max(width - 1, 1), yy * 255 // max(height - 1, 1), (xx ^ yy) % 256], -1)
    im = Image.fromarray(base.astype(np.uint8))
    d = ImageDraw.Draw(im)
    for _ in range(20):
        x0, y0 = int(rng.integers(0, width)), int(rng.integers(0, height))
        d.rectangle([x0, y0, x0 + int(rng.integers(8, width // 3)), y0 + int(rng.integers(8, height // 3))],
                    fill=tuple(int(c) for c in rng.integers(0, 256, 3)))
    for line in range(0, height, 24):
        d.text((8, line), f"line {line}: def f(x): return x * {line} + 1  # comment", fill=(0, 0, 0))
    return im


def bf16_to_f32(raw):
    return (raw.astype(np.uint32) << 16).view(np.float32)


def prepare(source, out):
    os.makedirs(out, exist_ok=True)
    path = source if source.endswith(".safetensors") else v.vision_file(source)
    weights = v.load_weights(v.read_safetensors(path))
    manifest = []
    for i, (name, width, height, reference) in enumerate(SIZES):
        rgb = v.resize_rgb(picture(width, height, i))
        h, w, _ = rgb.shape
        rgb.tofile(os.path.join(out, name + ".rgb"))
        entry = dict(name=name, height=h, width=w)
        ref_path = os.path.join(out, name + ".ref.npy")
        if reference and not os.path.exists(ref_path):
            patches, grid = v.patchify(rgb)
            np.save(ref_path, v.encode(weights, patches, grid).astype(np.float32))
        manifest.append(entry)
        print(json.dumps(entry), flush=True)
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)


def compare(out):
    results = []
    for entry in json.load(open(os.path.join(out, "manifest.json"))):
        ref_path = os.path.join(out, entry["name"] + ".ref.npy")
        dev_path = os.path.join(out, entry["name"] + ".rgb.bf16")
        if not os.path.exists(ref_path) or not os.path.exists(dev_path):
            continue
        ref = np.load(ref_path).astype(np.float64)
        dev = bf16_to_f32(np.fromfile(dev_path, dtype=np.uint16)).astype(np.float64).reshape(ref.shape)
        # bf16 rounding of the reference alone, for scale.
        rounded = bf16_to_f32(((ref.astype(np.float32).view(np.uint32) + 0x7FFF +
                                ((ref.astype(np.float32).view(np.uint32) >> 16) & 1)) >> 16).astype(np.uint16))
        row_cos = (ref * dev).sum(1) / np.linalg.norm(ref, axis=1) / np.linalg.norm(dev, axis=1)
        result = dict(name=entry["name"], tokens=ref.shape[0], max_abs=float(np.abs(dev - ref).max()),
                      ref_max=float(np.abs(ref).max()), rms_err=float(np.sqrt(((dev - ref) ** 2).mean())),
                      ref_rms=float(np.sqrt((ref ** 2).mean())),
                      bf16_round_max_abs=float(np.abs(rounded - ref).max()),
                      cosine=float((ref * dev).sum() / np.linalg.norm(ref) / np.linalg.norm(dev)),
                      min_token_cosine=float(row_cos.min()))
        results.append(result)
        print(json.dumps(result), flush=True)
    json.dump(results, open(os.path.join(out, "compare.json"), "w"), indent=1)


if __name__ == "__main__":
    {"prepare": lambda: prepare(sys.argv[2], sys.argv[3]), "compare": lambda: compare(sys.argv[2])}[sys.argv[1]]()
