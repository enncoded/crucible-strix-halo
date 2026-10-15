"""Content-addressed cache of named-baseline gate captures.

A baseline capture is bitwise deterministic for one binary, checkpoint, prompt,
command, environment and GPU software stack. The key covers all of these, so a
gate run can reuse a verified earlier capture instead of a GPU lease. Every
entry has a manifest of file hashes; a lookup re-hashes every file and refuses
an entry that does not verify (the caller then captures fresh).
"""
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import time

# Bump when capture() changes what a baseline capture produces or records.
# 2: a reference capture runs CRUCIBLE_PLE_HOST=0 when the candidate runs the host PLE gather.
# 3: also for the other values that the engine reads as 1 ('01', '+1', ' 1').
# 4: train 15's one capture environment: the CRUCIBLE_PLE_HOST rule (3), CRUCIBLE_EXACT=1 drops the numerics
#    switches from the reference (numerics-b, which named it capture-2 on its branch), and the key holds the
#    MTP sidecar's sha256, not its path (mtp-knob). The key also holds the source pin (key_inputs).
CAPTURE_VERSION = 'gates-baseline-capture-4'
# sha256 of the sources of gates.capture_command, capture_environment, cache_environment and baseline_options
# for each version (tests/test_gates_cache.py checks it; versions 1 to 3 pinned the sources without
# cache_environment). A source change needs a new version and pin.
CAPTURE_SOURCE_SHA256 = {'gates-baseline-capture-1': '2b7ba7683753d6d3cc64a3c651f762d54576e46b7faa62171ea2d068d9147bc2',
                         'gates-baseline-capture-2': '3184f922091264bcbd6c17994e2cee19a156d318a801b84a865cbe9775449136',
                         'gates-baseline-capture-3': 'af19fc72693ce364676601940703a9ba6b9e6f6656f191108ddd78a31ba38dcd',
                         'gates-baseline-capture-4': 'f626cda57b655616eb7f14dc3882effc2f998dd8131e9517c723f6cf305a1365'}
WIDTH = 4 * 2560
NAMES = tuple(f"h_{layer:02d}.f32" for layer in range(49))
COPIED = ('run.json', 'stdout.log', 'stderr.log', 'ids.txt')
# Names whose values can change what the engine computes. Values of names that
# are not public knobs go into the key only as hashes (they can hold credentials).
ENV_PREFIXES = ('CRUCIBLE_', 'HSA_', 'HIP_', 'ROCR_', 'ROCM_', 'GPU_', 'AMD_', 'HCC_', 'OMP_')
ENV_NAMES = ('LD_LIBRARY_PATH', 'LD_PRELOAD')
# The GPU queue's own settings (wait order, day override) never reach the engine's numerics.
ENV_IGNORED = ('GPU_RUN_',)
DRM = Path('/sys/class/drm')
KFD_NODES = Path('/sys/class/kfd/kfd/topology/nodes')
PCI_FIELDS = ('vendor', 'device', 'revision', 'subsystem_vendor', 'subsystem_device')
GFX_FIELDS = ('gfx_target_version', 'vendor_id', 'device_id', 'simd_count')
ROCM_VERSION = Path('/opt/rocm/.info/version')
READ_ONLY_FILE, READ_ONLY_DIR = 0o444, 0o555


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def write_atomic(path, data):
    temp = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}')
    temp.write_bytes(data)
    os.replace(temp, path)


def file_stamp(info):
    return [info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino]


class FileMemo:
    """sha256 per file, reused while path, size, mtime, ctime and inode match."""

    def __init__(self, cache):
        self.path = Path(cache) / 'file-hashes.json'
        try:
            self.known = json.loads(self.path.read_text())
        except (FileNotFoundError, ValueError):
            self.known = {}

    def sha256(self, path):
        real = os.path.realpath(path)
        info = os.stat(real)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f'not a regular file: {path}')
        stamp = file_stamp(info)
        entry = self.known.get(real)
        if entry and entry.get('stamp') == stamp:
            return info.st_size, entry['sha256']
        value = digest(real)
        if file_stamp(os.stat(real)) != stamp:
            raise ValueError(f'file changed while it was hashed: {path}')
        self.known = {**self.known, real: {'stamp': stamp, 'sha256': value}}
        self.save(real)
        return info.st_size, value

    def save(self, real):
        # Merge with entries that other gate runs wrote meanwhile; the last writer keeps all.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix('.lock'), 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                current = json.loads(self.path.read_text())
            except (FileNotFoundError, ValueError):
                current = {}
            write_atomic(self.path, canonical({**current, real: self.known[real]}))


def checkpoint_files(checkpoint, memo):
    files = []
    for directory, dirs, names in os.walk(checkpoint):
        dirs.sort()
        for name in sorted(names):
            path = Path(directory) / name
            size, sha = memo.sha256(path)
            files.append({'name': path.relative_to(checkpoint).as_posix(), 'bytes': size, 'sha256': sha})
    return sorted(files, key=lambda f: f['name'])


def output_environment(env, public=()):
    chosen = {k: v for k, v in env.items()
              if (k.startswith(ENV_PREFIXES) or k in ENV_NAMES) and not k.startswith(ENV_IGNORED)}
    return {k: v if k in public else 'sha256:' + hashlib.sha256(v.encode()).hexdigest()
            for k, v in sorted(chosen.items())}


def normalize_command(command, engine, checkpoint, ids, dump):
    names = {str(engine): '<engine>', str(checkpoint): '<checkpoint>', str(ids): '<ids>', str(dump): '<dump>'}
    return [names.get(str(word), str(word)) for word in command]


def libraries(binary, env, memo):
    # The loader resolves the HIP and HSA runtimes; their bytes identify the ROCm build.
    done = subprocess.run(['ldd', str(binary)], capture_output=True, text=True, env=env, timeout=30, check=False)
    if done.returncode:
        raise ValueError(f'ldd failed for {binary}: {done.stderr.strip()}')
    paths = sorted(set(re.findall(r'(?:=>\s*)?(/\S+)\s+\(0x', done.stdout)))
    return {path: memo.sha256(path)[1] for path in paths}


def device_identity():
    """PCI IDs of each GPU card and the gfx target of each KFD GPU node."""
    pci = {}
    for card in sorted(DRM.glob('card*/device/fw_version')):
        device = card.parent
        for name in PCI_FIELDS:
            if (device / name).exists():
                pci[f'{device.parent.name}/{name}'] = (device / name).read_text().strip()
    gfx = []
    for node in sorted(KFD_NODES.glob('*/properties')):
        props = dict(line.split(None, 1) for line in node.read_text().splitlines() if len(line.split()) == 2)
        if props.get('gfx_target_version', '0') != '0':
            gfx.append({k: props[k] for k in GFX_FIELDS if k in props})
    if not gfx:
        raise ValueError(f'no GPU gfx target under {KFD_NODES}; the cache key needs it')
    return {'pci': pci, 'gfx': gfx}


def platform_identity(binary, env, memo):
    firmware = {}
    for card in sorted(DRM.glob('card*/device/fw_version')):
        for path in sorted(card.iterdir()):
            firmware[f'{card.parent.parent.name}/{path.name}'] = path.read_text().strip()
        vbios = card.parent / 'vbios_version'
        if vbios.exists():
            firmware[f'{card.parent.parent.name}/vbios_version'] = vbios.read_text().strip()
    if not firmware:
        raise ValueError(f'no GPU firmware versions under {DRM}; the cache key needs them')
    uname = os.uname()
    return {'rocm_version': ROCM_VERSION.read_text().strip(), 'kernel': f'{uname.release} {uname.version}',
            'firmware': firmware, 'device': device_identity(), 'libraries': libraries(binary, env, memo)}


def key_inputs(*, binary_sha256, build_record_sha256, checkpoint, ids_sha256, command, environment, platform):
    # The source pin keys the entry too: an entry names the capture code that made it, not only its version.
    return {'capture_version': CAPTURE_VERSION, 'capture_sources_sha256': CAPTURE_SOURCE_SHA256[CAPTURE_VERSION],
            'binary_sha256': binary_sha256,
            'build_record_sha256': build_record_sha256, 'checkpoint': checkpoint, 'ids_sha256': ids_sha256,
            'command': list(command), 'environment': environment, 'platform': platform}


def cache_key(inputs):
    return hashlib.sha256(canonical({**inputs, 'capture_version': CAPTURE_VERSION})).hexdigest()


def entries(cache):
    path = Path(cache) / 'entries'
    path.mkdir(parents=True, exist_ok=True)
    return path


def tree(entry):
    files = {}
    for directory, dirs, names in os.walk(entry):
        for name in [*dirs, *names]:
            path = Path(directory) / name
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError(f'not a private regular file: {path.relative_to(entry)}')
            files[path.relative_to(entry).as_posix()] = info.st_size
    return files


def verify(entry, key, inputs, rows):
    """Re-hash every file of an entry against its manifest. Raises ValueError."""
    if entry.is_symlink() or not entry.is_dir():
        raise ValueError('entry is not a directory')
    try:
        manifest = json.loads((entry / 'manifest.json').read_text())
        listed = manifest['files']
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f'unreadable manifest: {error}') from error
    if manifest.get('key') != key or manifest.get('schema') != 1:
        raise ValueError('manifest key differs')
    found = tree(entry)
    found.pop('manifest.json', None)
    if set(found) != set(listed):
        raise ValueError(f'file set differs from the manifest: {sorted(set(found) ^ set(listed))}')
    expected = rows * WIDTH * 4
    for name, meta in sorted(listed.items()):
        if found[name] != meta['bytes'] or (name.startswith('dump/') and meta['bytes'] != expected):
            raise ValueError(f'size differs: {name}')
        if digest(entry / name) != meta['sha256']:
            raise ValueError(f'sha256 differs: {name}')
    if set(listed) != {*COPIED, 'key.json', 'provenance.json', *(f'dump/{n}' for n in NAMES)}:
        raise ValueError('entry does not hold a complete capture')
    stored = json.loads((entry / 'key.json').read_text()).get('inputs')
    if not isinstance(stored, dict) or cache_key(stored) != key or stored != inputs:
        raise ValueError('stored key inputs differ from this capture')
    ids = (entry / 'ids.txt').read_text().strip()
    return {'entry': str(entry), 'key': key,
            'manifest_sha256': digest(entry / 'manifest.json'),
            'provenance': json.loads((entry / 'provenance.json').read_text()),
            'run': json.loads((entry / 'run.json').read_text()),
            'output_ids': [int(value) for value in ids.split(',')],
            'dumps': {name: listed[f'dump/{name}']['sha256'] for name in NAMES}}


def retire(cache, key, reason):
    """Move an entry aside so the key is free; keep it for inspection."""
    entry = entries(cache) / key
    target = entry.with_name(f'.bad.{key}.{time.time_ns()}')
    try:
        os.rename(entry, target)
    except FileNotFoundError:
        return None
    try:
        (target.parent / f'{target.name}.reason').write_text(reason + '\n')
    except OSError:
        pass
    return target


def lookup(cache, key, inputs, rows):
    """(verified record, None), (None, None) on a miss, or (None, reason) for a refused entry."""
    entry = entries(cache) / key
    if not os.path.lexists(entry):
        return None, None
    try:
        return verify(entry, key, inputs, rows), None
    except (OSError, ValueError, KeyError, TypeError) as error:
        reason = f'refused cache entry {entry}: {error}'
        retire(cache, key, reason)
        return None, reason


def freeze(root):
    for directory, dirs, names in os.walk(root, topdown=False):
        for name in names:
            os.chmod(Path(directory) / name, READ_ONLY_FILE)
        os.chmod(directory, READ_ONLY_DIR)


def thaw_remove(root):
    for directory, dirs, names in os.walk(root):
        os.chmod(directory, 0o755)
    shutil.rmtree(root)


def store(cache, key, inputs, capture_dir, record, provenance, check_dumps=True):
    """Copy a checked capture into the cache atomically. Returns the entry, or
    None when another run filled the key first."""
    root = entries(cache)
    temp = root / f'.tmp.{key}.{os.getpid()}.{time.time_ns()}'
    try:
        (temp / 'dump').mkdir(parents=True)
        for name in COPIED:
            shutil.copyfile(capture_dir / name, temp / name)
        for name in NAMES:
            shutil.copyfile(capture_dir / 'dump' / name, temp / 'dump' / name)
        (temp / 'key.json').write_bytes(canonical({'key': key, 'inputs': inputs}))
        (temp / 'provenance.json').write_bytes(canonical(provenance))
        files = {name: {'bytes': size, 'sha256': digest(temp / name)} for name, size in sorted(tree(temp).items())}
        # The cache must hold the bytes that the gate checked, not a later change.
        if check_dumps and any(files[f'dump/{n}']['sha256'] != record['dumps'][n]['sha256'] for n in NAMES):
            raise ValueError('cache copy differs from the checked capture')
        (temp / 'manifest.json').write_bytes(canonical({'schema': 1, 'key': key, 'files': files}))
        freeze(temp)
        try:
            os.rename(temp, root / key)
        except OSError as error:
            if error.errno in (errno.EEXIST, errno.ENOTEMPTY):
                thaw_remove(temp)
                return None
            raise
        return root / key
    except BaseException:
        if temp.exists():
            thaw_remove(temp)
        raise


def record_use(cache, key, every, note):
    """Count reuses of one key; every Nth (starting with the first) is due for a fresh recheck."""
    path = Path(cache) / 'uses' / f'{key}.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as log:
        fcntl.flock(log, fcntl.LOCK_EX)
        log.seek(0)
        index = sum(1 for _ in log)
        log.write(json.dumps({'ns': time.time_ns(), **note}) + '\n')
    return {'every': every, 'index': index, 'due': every > 0 and index % every == 0}


def provenance(record, gate_output, gate_tool_sha256):
    # Host, user, pid and absolute paths are fine on the test machine. Remove them before a report.json or
    # cache entry is copied into docs/ or anywhere public.
    return {'started_ns': record['started_ns'], 'ended_ns': record['ended_ns'], 'host': socket.gethostname(),
            'user': os.environ.get('USER'), 'pid': os.getpid(), 'gate_output': str(gate_output),
            'gate_tool_sha256': gate_tool_sha256, 'lease': {'queued_command': record['queued_command'][:6],
                                                            'returncode': record['returncode']}}
