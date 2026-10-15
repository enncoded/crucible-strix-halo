"""Module map and include rules for engine/ and kernels/.

tools/layout_map.tsv gives every engine and kernel file one module and one
target path. The rules here read the map, so they give the same answer before
and after a layout move: a path is first made canonical
(its target path), then checked by module.
"""
from __future__ import annotations

import fnmatch
import functools
import posixpath
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = 'tools/layout_map.tsv'
MAP_HEADER = ('old_path', 'new_path', 'module', 'reason', 'decide')
SOURCE_DIRS = ('engine/', 'kernels/', 'bench/kernels/')
SOURCE_SUFFIXES = ('.h', '.hpp', '.hip', '.inc', '.cpp', '.cc', '.c', '.cuh')
# Includes go down only: a module may include its own rank or a lower rank.
RANKS = {
    'gfx1151': 0,
    'exl3': 1, 'exl3/kernels': 1,
    'layers': 2, 'layers/kernels': 2,
    'model/flashnext': 3,
    'runtime': 4,
    'serve': 5,
    'cli': 6,
    'engine': 7,  # engine/engine.hip, the one translation unit
    'bench': 8,  # bench/kernels: bench-only kernel headers; engine code never includes them
}
# Leaves: any module (kernels too) may include them, and they include nothing
# quoted. Fixed shapes need the named model geometry at every rank (an earlier change grows
# it into the model config); every switch read goes through the registry;
# image and state digests use SHA-256 (design decision, Oct 10).
LEAVES = frozenset({'engine/model/flashnext/geometry.h', 'engine/switches.h',
                    'engine/runtime/sha256.h'})
INCLUDE_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"]+)"', re.M)


@dataclass(frozen=True)
class Row:
    old: str
    new: str
    module: str
    reason: str
    decide: str


MODULE_DIRS = {'engine': 'engine', 'bench': 'bench/kernels'}


def module_dir(module: str) -> str:
    return MODULE_DIRS.get(module, f'engine/{module}')


def is_kernel_module(module: str) -> bool:
    return module.endswith('/kernels')


def load_map(root: Path = ROOT) -> tuple[Row, ...]:
    lines = (root / MAP_PATH).read_text(encoding='utf-8').splitlines()
    if not lines or tuple(lines[0].split('\t')) != MAP_HEADER:
        raise ValueError(f'{MAP_PATH}: header must be {MAP_HEADER}')
    rows = []
    for number, line in enumerate(lines[1:], start=2):
        fields = line.split('\t')
        if len(fields) != len(MAP_HEADER):
            raise ValueError(f'{MAP_PATH}:{number}: expected 5 tab-separated fields')
        row = Row(*fields)
        if row.module not in RANKS:
            raise ValueError(f'{MAP_PATH}:{number}: unknown module {row.module!r}')
        if posixpath.dirname(row.new) != module_dir(row.module):
            raise ValueError(f'{MAP_PATH}:{number}: {row.new} is not in {module_dir(row.module)}/')
        rows.append(row)
    return tuple(rows)


class Layout:
    """Lookups from a current path (old or new) to its map row."""

    def __init__(self, rows: tuple[Row, ...]):
        self.rows = rows
        self.by_old = {r.old: r for r in rows}
        self.by_new = {r.new: r for r in rows}

    def row(self, path: str) -> Row | None:
        return self.by_new.get(path) or self.by_old.get(path)

    def canonical(self, path: str) -> str | None:
        row = self.row(path)
        return row.new if row else None


def tracked_files(root: Path = ROOT) -> frozenset[str]:
    """Files in the git index. Untracked files must not hide a clean-clone fault.

    A tree without git (a `git archive` export, as the train's CPU checks use)
    has no untracked files, so a walk that skips .gitignore entries is the same.
    """
    if not (root / '.git').exists():
        return walked_files(root)
    out = subprocess.run(['git', 'ls-files', '-z'], cwd=root, check=True,
                         capture_output=True, text=True).stdout
    return frozenset(p for p in out.split('\0') if p)


def walked_files(root: Path) -> frozenset[str]:
    ignore = root / '.gitignore'
    patterns = [line.strip().strip('/') for line in
                (ignore.read_text().splitlines() if ignore.exists() else [])
                if line.strip() and not line.startswith('#')]

    def ignored(rel: str) -> bool:
        parts = rel.split('/')
        return any(fnmatch.fnmatch(part, p) for p in patterns if '/' not in p for part in parts) \
            or any(rel == p or rel.startswith(p + '/') for p in patterns if '/' in p)

    return frozenset(rel for rel in (p.relative_to(root).as_posix() for p in root.rglob('*')
                                     if p.is_file()) if not ignored(rel))


def is_source(path: str) -> bool:
    return path.startswith(SOURCE_DIRS) and path.endswith(SOURCE_SUFFIXES)


def quoted_includes(text: str) -> list[str]:
    return INCLUDE_RE.findall(text)


@functools.lru_cache(maxsize=None)
def _repo_layout() -> Layout:
    return Layout(load_map())


def source_key(path: str, lay: Layout | None = None) -> str:
    """The target path of PATH, so that a build record written before the move
    (keys like engine/gen_cli.h) and one written after it compare equal."""
    row = (lay or _repo_layout()).row(path)
    return row.new if row else path


def source_keys(paths, lay: Layout | None = None) -> set[str]:
    return {source_key(p, lay) for p in paths}


def resolve(src: str, include: str) -> str:
    return posixpath.normpath(posixpath.join(posixpath.dirname(src), include))


def include_rule(src: Row, dst: Row) -> str | None:
    """The broken rule's name, or None when src may include dst."""
    if src.new in LEAVES:
        return 'leaf'
    if dst.new in LEAVES:
        return None
    if is_kernel_module(src.module) and not (
            is_kernel_module(dst.module) or dst.module == 'gfx1151'):
        return 'kernel'
    if RANKS[dst.module] > RANKS[src.module]:
        return 'upward'
    return None


def violations(root: Path = ROOT, layout: Layout | None = None,
               tracked: frozenset[str] | None = None) -> list[str]:
    """Every broken include rule, as 'rule: src -> dst' with canonical paths.

    missing: the target is not a tracked file (the include text is kept, so
    the key does not change when the file moves). outside: an engine or kernel
    file includes bench/, tests/ or another tree.
    """
    layout = layout or Layout(load_map(root))
    tracked = tracked if tracked is not None else tracked_files(root)
    found = []
    for path in sorted(p for p in tracked if is_source(p)):
        src = layout.row(path)
        if src is None:
            continue  # the ownership test reports files with no map row
        text = (root / path).read_text(encoding='utf-8', errors='replace')
        for include in quoted_includes(text):
            target = resolve(path, include)
            dst = layout.row(target)
            if target not in tracked:
                found.append(f'missing: {src.new} -> "{include}"')
            elif dst is None:
                if not is_source(target):
                    found.append(f'outside: {src.new} -> {target}')
            else:
                rule = include_rule(src, dst)
                if rule:
                    found.append(f'{rule}: {src.new} -> {dst.new}')
    return sorted(set(found))
