"""Strict HTTP Range safetensors reader. Never falls back to full shard downloads."""
from __future__ import annotations
import json
import math
import struct
import urllib.request
from pathlib import Path
import numpy as np

DTYPES = {"I16": "<i2", "I32": "<i4", "F16": "<f2", "F32": "<f4", "BF16": "<u2"}


def read_range(url: str, start: int, end: int) -> bytes:
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
        raise ValueError('Invalid byte range')
    if end - start + 1 > 128_000_000:
        raise ValueError('Refusing payload range above 128 MB')
    # A unique query avoids CDN caches incorrectly serving another requested range.
    query = "&" if "?" in url else "?"
    request = urllib.request.Request(f"{url}{query}range={start}-{end}",
                                    headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError(f"Range rejected: HTTP {response.status}")
        expected = f"bytes {start}-{end}/"
        if not response.headers.get("Content-Range", "").startswith(expected):
            raise RuntimeError("Unexpected Content-Range")
        body = response.read(end - start + 2)
    if len(body) != end - start + 1:
        raise RuntimeError("Incorrect ranged response length")
    return body


class SafeRanges:
    def __init__(self, repo: str, revision: str, shard: str, cache: Path):
        self.url = f"https://huggingface.co/{repo}/resolve/{revision}/{shard}"
        self.repo, self.revision, self.shard = repo, revision, shard
        cache.mkdir(parents=True, exist_ok=True)
        header_file = cache / (repo.replace("/", "_") + "_" + revision.replace("/", "_") + "_" + shard + ".header.json")
        if header_file.exists():
            record = json.loads(header_file.read_text())
            self.base, self.header = record["base"], record["header"]
        else:
            length = struct.unpack("<Q", read_range(self.url, 0, 7))[0]
            if not 2 <= length <= 100_000_000:
                raise RuntimeError("Unreasonable safetensors header length")
            self.base = 8 + length
            self.header = json.loads(read_range(self.url, 8, self.base - 1))
            header_file.write_text(json.dumps({"base": self.base, "header": self.header}, indent=2))

    def tensor(self, name: str, expert: int | None = None) -> np.ndarray:
        info = self.header[name]
        shape = tuple(info["shape"])
        start, end = info["data_offsets"]
        if info['dtype'] not in DTYPES or any(not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError('Invalid tensor dtype or shape')
        dtype = np.dtype(DTYPES[info["dtype"]])
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end - start != math.prod(shape) * dtype.itemsize:
            raise ValueError('Tensor byte offsets disagree with shape')
        if expert is not None:
            if not 0 <= expert < shape[0]:
                raise ValueError("Expert index out of range")
            size = math.prod(shape[1:]) * dtype.itemsize
            start, end = start + expert * size, start + (expert + 1) * size
            shape = shape[1:]
        raw = read_range(self.url, self.base + start, self.base + end - 1)
        array = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        if info["dtype"] == "BF16":
            array = (array.astype(np.uint32) << 16).view(np.float32)
        return array
