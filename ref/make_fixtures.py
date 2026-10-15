"""Rebuild full-matrix fixture oracles using bounded HTTP ranges, no GPU."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from checkpoint_ranges import SafeRanges
from exl3_ref import reconstruct

QUANT_REPO = 'turboderp/Qwen3.8-Flash-Next-exl3'
QUANT_REVISION = '69e33439ae950f17bcbe95c98f117d80f759ab6d'
ORIGINAL_REPO = 'Qwen/Qwen3.8-Flash-Next'
ORIGINAL_REVISION = 'de4b8e4d43b917e7706784d8bb445c9af86a3540'
LAYERS = [
    ('layer0_qkv', 'model.language_model.layers.0.linear_attn.in_proj_qkv',
     'model-00001-of-00131.safetensors', 'model.language_model.layers.0.linear_attn.in_proj_qkv.weight', None),
    ('layer0_expert0_down', 'model.language_model.layers.0.mlp.experts.0.down_proj',
     'model-00003-of-00131.safetensors', 'model.language_model.layers.0.mlp.experts.down_proj', 0),
]


def save_npz(path: Path, arrays: dict):
    # np.load understands BZIP2 zip members; it saves about 1.3 MB versus deflate here.
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_BZIP2, compresslevel=9) as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(array), allow_pickle=False)
            member = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_BZIP2
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue(), compresslevel=9)


def metrics(reconstructed: np.ndarray, original: np.ndarray) -> tuple[float, float]:
    a, b = reconstructed.astype(np.float64).ravel(), original.astype(np.float64).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b)), float(np.corrcoef(a, b)[0, 1])


def main():
    out = Path(__file__).parent / 'fixtures'
    out.mkdir(exist_ok=True)
    cache = Path('/tmp/crucible-data')
    quant = SafeRanges(QUANT_REPO, QUANT_REVISION, 'model-00001-of-00007.safetensors', cache)
    records = []
    for tag, name, shard, original_name, expert in LAYERS:
        stored = {suffix: quant.tensor(name + '.' + suffix) for suffix in ['trellis', 'suh', 'svh', 'mul1']}
        w = reconstruct(stored)
        original = SafeRanges(ORIGINAL_REPO, ORIGINAL_REVISION, shard, cache).tensor(original_name, expert).T
        error, correlation = metrics(w, original)
        save_npz(out / (tag + '.npz'), dict(stored, W=w.astype(np.float16),
                 relative_frobenius_error=error, correlation=correlation,
                 original_sha256=hashlib.sha256(original.T.astype('<f4').tobytes()).hexdigest(),
                 tensor_name=name))
        record = dict(name=name, fixture=tag + '.npz', shape=list(w.shape), K=stored['trellis'].shape[-1] / 16,
                      relative_frobenius_error=error, correlation=correlation,
                      fixture_bytes=(out / (tag + '.npz')).stat().st_size,
                      original_repo=ORIGINAL_REPO, original_revision=ORIGINAL_REVISION,
                      original_shard=shard, original_name=original_name, original_expert=expert)
        records.append(record)
        print(json.dumps(record), flush=True)
    (out / 'validation.json').write_text(json.dumps(dict(quant_repo=QUANT_REPO,
        quant_revision=QUANT_REVISION, layers=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
