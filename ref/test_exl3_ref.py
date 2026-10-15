"""Offline tests; set EXLLAMAV3_SRC to run upstream CPU-only function checks."""
from pathlib import Path
import ast
import math
import os
import numpy as np
import pytest
from exl3_ref import decode_tile_symbols, decode_codebook, reconstruct, matvec, tensor_core_perm

FIXTURES = Path(__file__).parent / "fixtures"


def pack_symbols(symbols, bits):
    widths = np.full(256, int(bits)) + (np.arange(256) % 2 if bits % 1 else 0)
    stream = ''.join(format(int(s), f'0{int(w)}b') for s, w in zip(symbols, widths))
    words = np.array([int(stream[i:i + 32], 2) for i in range(0, len(stream), 32)], dtype='<u4')
    return words.view('<i2')


@pytest.mark.parametrize('bits', [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8])
def test_ring_decode(bits):
    rng = np.random.default_rng(91)
    widths = np.full(256, int(bits)) + (np.arange(256) % 2 if bits % 1 else 0)
    symbols = np.array([rng.integers(1 << int(w)) for w in widths])
    stream = ''.join(format(int(s), f'0{int(w)}b') for s, w in zip(symbols, widths))
    endpoints = np.cumsum(widths)
    expected = np.array([int(''.join(stream[(int(e) - 16 + j) % len(stream)] for j in range(16)), 2)
                         for e in endpoints], dtype=np.uint16)
    decoded = decode_tile_symbols(pack_symbols(symbols, bits), bits)
    np.testing.assert_array_equal(decoded, expected)
    np.testing.assert_array_equal(decoded & ((1 << widths) - 1), symbols)


def test_validation():
    with pytest.raises(ValueError):
        decode_tile_symbols(np.zeros(48, dtype=np.int16), 4)
    with pytest.raises(ValueError):
        decode_tile_symbols(np.zeros(48, dtype=np.int16), 3.2)
    assert len(np.unique(tensor_core_perm())) == 256
    for bits in [0, 9, 4.5, np.nan]:
        with pytest.raises(ValueError):
            decode_tile_symbols(np.zeros(48, dtype=np.int16), bits)
    with pytest.raises(ValueError):
        decode_tile_symbols(np.zeros(48, dtype=np.float32))


@pytest.mark.parametrize('codebook', ['mcg', '3inst'])
def test_other_codebooks_scalar(codebook):
    import struct
    def half(bits):
        return struct.unpack('<e', struct.pack('<H', bits))[0]
    states = np.arange(65536, dtype=np.uint16)
    expected = []
    for state in states:
        product = (int(state) * (0xCBAC1FED if codebook == 'mcg' else 89226354) +
                   (0 if codebook == 'mcg' else 64248484)) & 0xffffffff
        encoded = (product & 0x8fff8fff) ^ 0x3b603b60
        expected.append(half(encoded & 65535) + half(encoded >> 16))
    np.testing.assert_array_equal(decode_codebook(states, codebook), np.array(expected, dtype=np.float16))
    with pytest.raises(ValueError):
        decode_codebook(states, 'unknown')


def upstream_functions(relative_path, names, env):
    source = Path(os.environ.get('EXLLAMAV3_SRC', '/tmp/crucible-exllamav3')) / relative_path
    if not source.exists():
        pytest.skip('Set EXLLAMAV3_SRC to the d3739fd ExLlamaV3 clone')
    tree = ast.parse(source.read_text())
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in body} == set(names)
    # Run unchanged function bodies without importing the CUDA-only package initialization.
    for node in body:
        node.decorator_list = []
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), env)
    return env


def test_upstream_cpu_codebook_and_permutation():
    torch = pytest.importorskip('torch')
    env = upstream_functions('exllamav3/modules/quant/exl3_lib/ngram_codec.py',
                             ['mul1_codebook'], {'torch': torch, 'MUL1': 0x83DCD12D})
    expected = env['mul1_codebook']('cpu').numpy()
    actual = decode_codebook(np.arange(65536, dtype=np.uint16), 'mul1')
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    env = upstream_functions('exllamav3/modules/quant/exl3_lib/quantize.py',
                             ['tensor_core_perm'], {'torch': torch})
    np.testing.assert_array_equal(tensor_core_perm(), env['tensor_core_perm']('cpu').numpy())


def torch_hadamard_functions():
    torch = pytest.importorskip('torch')
    env = upstream_functions('exllamav3/util/hadamard.py', ['sylvester', 'get_hadamard_dt'], {'torch': torch})
    h = torch.ones((1, 1), dtype=torch.float16)
    for _ in range(7):
        h = env['sylvester'](h)
    env['get_hadamard'] = lambda n: h if n == 128 else None
    return upstream_functions('exllamav3/modules/quant/exl3_lib/quantize.py',
                              ['preapply_had_l', 'preapply_had_r'],
                              dict(env, math=math)), torch


@pytest.mark.parametrize('filename', ['layer0_qkv.npz', 'layer0_expert0_down.npz'])
def test_real_fixture(filename):
    fixture = FIXTURES / filename
    if not fixture.exists():
        pytest.skip('Run .venv/bin/python ref/make_fixtures.py to generate ignored fixtures')
    with np.load(fixture) as data:
        stored = {key: data[key] for key in ['trellis', 'suh', 'svh', 'mul1']}
        expected = data['W'].astype(np.float32)
        actual = reconstruct(stored)
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (stored['suh'].size, stored['svh'].size)
        assert float(data['relative_frobenius_error']) < 0.25
        assert float(data['correlation']) > 0.95


@pytest.mark.parametrize('filename', ['layer0_qkv.npz', 'layer0_expert0_down.npz'])
def test_upstream_cpu_reconstruct(filename):
    from exl3_ref import reconstruct_inner
    if not (FIXTURES / filename).exists():
        pytest.skip('Run .venv/bin/python ref/make_fixtures.py to generate ignored fixtures')
    env, torch = torch_hadamard_functions()
    with np.load(FIXTURES / filename) as data:
        stored = {key: data[key] for key in ['trellis', 'suh', 'svh', 'mul1']}
    inner = torch.from_numpy(reconstruct_inner(stored))
    expected = env['preapply_had_l'](inner, 128)
    expected = expected * torch.from_numpy(stored['suh'])[:, None]
    expected = env['preapply_had_r'](expected, 128)
    expected = expected * torch.from_numpy(stored['svh'])[None, :]
    actual = reconstruct(stored)
    np.testing.assert_array_equal(actual, expected.numpy().astype(np.float32))
    diff = np.abs(actual - expected.numpy().astype(np.float32))
    print(f'upstream CPU reconstruct max_abs={diff.max():.9g} unequal={np.count_nonzero(diff)}')


def test_matvec_operation_order():
    from exl3_ref import reconstruct_inner, hadamard_kernel
    if not (FIXTURES / 'layer0_expert0_down.npz').exists():
        pytest.skip('Run .venv/bin/python ref/make_fixtures.py to generate ignored fixtures')
    with np.load(FIXTURES / 'layer0_expert0_down.npz') as data:
        stored = {key: data[key] for key in ['trellis', 'suh', 'svh', 'mul1']}
    x = np.random.default_rng(2).normal(size=640).astype(np.float16)
    xh = hadamard_kernel(x, pre_scale=stored['suh'])
    yh = (xh.astype(np.float32) @ reconstruct_inner(stored).astype(np.float32)).astype(np.float16)
    expected = hadamard_kernel(yh, post_scale=stored['svh'])
    np.testing.assert_array_equal(matvec(x, stored), expected)
    rotated32 = xh.astype(np.float32) @ reconstruct_inner(stored).astype(np.float32)
    expected32 = hadamard_kernel(rotated32, post_scale=stored['svh'])
    np.testing.assert_array_equal(matvec(x, stored, np.float32), expected32)
    with_bias = dict(stored, bias=np.ones(2560, dtype=np.float16))
    np.testing.assert_array_equal(matvec(x, with_bias), (expected.astype(np.float32) + 1).astype(np.float16))
    with pytest.raises(ValueError):
        matvec(x[:-1], stored)
    with pytest.raises(ValueError):
        matvec(x, stored, np.float64)


def test_range_reader_rejects_full_download(monkeypatch):
    from checkpoint_ranges import read_range
    import urllib.request
    class Response:
        status = 200
        headers = {}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, size):
            raise AssertionError('Must reject response before reading body')
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *args, **kwargs: Response())
    with pytest.raises(RuntimeError, match='Range rejected'):
        read_range('https://example.test/shard', 0, 7)


def test_range_reader_expert_slice(tmp_path, monkeypatch):
    import checkpoint_ranges as ranges
    import json
    import struct
    header = {'experts': {'dtype': 'BF16', 'shape': [2, 2, 2], 'data_offsets': [0, 16]}}
    encoded = json.dumps(header).encode()
    blob = struct.pack('<Q', len(encoded)) + encoded + np.array([0, 0, 0, 0, 0x3f80, 0x4000, 0x4040, 0x4080], dtype='<u2').tobytes()
    requested = []
    def read(url, start, end):
        requested.append((start, end))
        return blob[start:end + 1]
    monkeypatch.setattr(ranges, 'read_range', read)
    safe = ranges.SafeRanges('test/repo', 'revision', 'shard.safetensors', tmp_path)
    np.testing.assert_array_equal(safe.tensor('experts', expert=1), [[1, 2], [3, 4]])
    assert requested[-1] == (8 + len(encoded) + 8, 8 + len(encoded) + 15)
    with pytest.raises(ValueError):
        safe.tensor('experts', expert=2)
