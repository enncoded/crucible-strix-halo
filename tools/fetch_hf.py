#!/usr/bin/env python3
"""Download one revision of a Hugging Face repo with resume and LFS sha256 checks.

Usage: fetch_hf.py REPO REVISION DEST [JOBS]
Stdlib + curl only, so it runs on a bare host.
"""
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import urllib.request


def tree(repo, rev):
    url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=1"
    with urllib.request.urlopen(url, timeout=60) as r:
        return [f for f in json.load(r) if f["type"] == "file"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(repo, rev, dest, f):
    name = f["path"]
    path = os.path.join(dest, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    want = f.get("lfs", {}).get("oid")
    size = f["size"]
    if not (os.path.exists(path) and os.path.getsize(path) == size):
        url = f"https://huggingface.co/{repo}/resolve/{rev}/{name}"
        subprocess.run(["curl", "-sSL", "--fail", "-C", "-", "--retry", "20",
                        "--retry-all-errors", "-o", path, url], check=True)
    if os.path.getsize(path) != size:
        raise RuntimeError(f"{name}: size {os.path.getsize(path)} != {size}")
    if want and sha256(path) != want:
        raise RuntimeError(f"{name}: sha256 mismatch")
    return name, size


def main():
    repo, rev, dest = sys.argv[1:4]
    jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    files = tree(repo, rev)
    total = sum(f["size"] for f in files)
    print(f"{repo}@{rev}: {len(files)} files, {total / 1e9:.1f} GB -> {dest}", flush=True)
    done = 0
    with cf.ThreadPoolExecutor(jobs) as ex:
        for name, size in ex.map(lambda f: fetch(repo, rev, dest, f), files):
            done += size
            print(f"ok {name} ({done / 1e9:.1f}/{total / 1e9:.1f} GB)", flush=True)
    print("ALL OK", flush=True)


if __name__ == "__main__":
    main()
