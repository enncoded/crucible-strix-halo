"""CPU tests for the content-addressed baseline capture cache of the gate tool."""
import json
import multiprocessing
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import baseline_cache as bc
import gates as g

PLATFORM = {'rocm_version': '7.2.4', 'kernel': '6.0', 'firmware': {'mec': '0x24'}, 'libraries': {}}


def inputs(**changes):
    base = dict(binary_sha256='a' * 64, build_record_sha256='b' * 64,
                checkpoint=[{'name': 'config.json', 'bytes': 2, 'sha256': 'c' * 64}],
                ids_sha256='d' * 64, command=['<engine>', 'gen', '<checkpoint>', '--n', '1'],
                environment={'CRUCIBLE_FUSE': '1'}, platform=PLATFORM)
    return bc.key_inputs(**{**base, **changes})


class KeyTests(unittest.TestCase):
    def test_every_input_changes_the_key(self):
        key = bc.cache_key(inputs())
        self.assertEqual(key, bc.cache_key(inputs()))
        self.assertRegex(key, r'^[0-9a-f]{64}$')
        changed = {
            'binary': inputs(binary_sha256='e' * 64),
            'build record': inputs(build_record_sha256='f' * 64),
            'no build record': inputs(build_record_sha256=None),
            'checkpoint name': inputs(checkpoint=[{'name': 'x.json', 'bytes': 2, 'sha256': 'c' * 64}]),
            'checkpoint size': inputs(checkpoint=[{'name': 'config.json', 'bytes': 3, 'sha256': 'c' * 64}]),
            'checkpoint sha': inputs(checkpoint=[{'name': 'config.json', 'bytes': 2, 'sha256': '0' * 64}]),
            'ids': inputs(ids_sha256='0' * 64),
            'argument': inputs(command=['<engine>', 'gen', '<checkpoint>', '--n', '2']),
            'environment value': inputs(environment={'CRUCIBLE_FUSE': '0'}),
            'environment name': inputs(environment={'CRUCIBLE_FUSE': '1', 'CRUCIBLE_NX': '1'}),
            'rocm': inputs(platform={**PLATFORM, 'rocm_version': '7.2.5'}),
            'kernel driver': inputs(platform={**PLATFORM, 'kernel': '6.1'}),
            'firmware': inputs(platform={**PLATFORM, 'firmware': {'mec': '0x25'}}),
        }
        keys = {name: bc.cache_key(value) for name, value in changed.items()}
        for name, value in keys.items():
            with self.subTest(name=name):
                self.assertNotEqual(value, key)
        self.assertEqual(len(set(keys.values())), len(keys))
        pin = bc.CAPTURE_SOURCE_SHA256[bc.CAPTURE_VERSION]
        with patch.object(bc, 'CAPTURE_VERSION', 'gates-capture-test'), \
                patch.dict(bc.CAPTURE_SOURCE_SHA256, {'gates-capture-test': pin}):
            self.assertNotEqual(bc.cache_key(inputs()), key)
        with patch.dict(bc.CAPTURE_SOURCE_SHA256, {bc.CAPTURE_VERSION: '0' * 64}):
            self.assertNotEqual(bc.cache_key(inputs()), key)

    def test_output_environment_keeps_runtime_names_and_hides_other_values(self):
        env = {'CRUCIBLE_FUSE': '1', 'HSA_USERPTR_FOR_PAGED_MEM': '0', 'HIP_VISIBLE_DEVICES': '0',
               'LD_LIBRARY_PATH': '/opt/x', 'HOME': '/srv/u', 'PWD': '/tmp', 'CRUCIBLE_TOKEN_X': 'secret',
               'GPU_RUN_DAY_OVERRIDE': 'reason', 'GPU_MAX_HW_QUEUES': '2'}
        out = bc.output_environment(env, public=('CRUCIBLE_FUSE',))
        self.assertNotIn('HOME', out)
        self.assertNotIn('PWD', out)
        self.assertEqual(out['CRUCIBLE_FUSE'], '1')
        # A value that is not a known knob goes into the key only as a hash.
        self.assertNotIn('secret', json.dumps(out))
        self.assertIn('CRUCIBLE_TOKEN_X', out)
        self.assertEqual(set(out), {'CRUCIBLE_FUSE', 'HSA_USERPTR_FOR_PAGED_MEM', 'HIP_VISIBLE_DEVICES',
                                    'LD_LIBRARY_PATH', 'CRUCIBLE_TOKEN_X', 'GPU_MAX_HW_QUEUES'})
        self.assertNotEqual(bc.output_environment({**env, 'CRUCIBLE_TOKEN_X': 'other'}), out)

    def test_command_paths_become_placeholders(self):
        command = ['/a/engine', 'gen', '/m/ckpt', '--ids-file', '/p/ids', '--dump', '/out/baseline/dump', '--n', '1']
        normal = bc.normalize_command(command, engine='/a/engine', checkpoint='/m/ckpt', ids='/p/ids',
                                      dump='/out/baseline/dump')
        self.assertEqual(normal, ['<engine>', 'gen', '<checkpoint>', '--ids-file', '<ids>', '--dump', '<dump>',
                                  '--n', '1'])


class PinTests(unittest.TestCase):
    def test_capture_sources_are_pinned_to_the_capture_version(self):
        import hashlib, inspect
        functions = (g.capture_command, g.capture_environment, g.cache_environment, g.baseline_options)
        source = ''.join(inspect.getsource(f) for f in functions)
        digest = hashlib.sha256(source.encode()).hexdigest()
        # A change to what a baseline capture runs needs a new CAPTURE_VERSION and a new pin.
        self.assertEqual(list(bc.CAPTURE_SOURCE_SHA256)[-1], bc.CAPTURE_VERSION)
        self.assertEqual(len(set(bc.CAPTURE_SOURCE_SHA256.values())), len(bc.CAPTURE_SOURCE_SHA256))
        self.assertEqual(bc.CAPTURE_SOURCE_SHA256[bc.CAPTURE_VERSION], digest,
                         f'capture sources changed: bump CAPTURE_VERSION and pin {digest}')

    def test_the_key_holds_the_source_pin(self):
        key = inputs()
        self.assertEqual(key['capture_sources_sha256'], bc.CAPTURE_SOURCE_SHA256[bc.CAPTURE_VERSION])
        self.assertEqual(key['capture_version'], 'gates-baseline-capture-4')


class DeviceIdentityTests(unittest.TestCase):
    def sysfs(self, root, gfx='110501'):
        card = root / 'drm/card1/device'
        (card / 'fw_version').mkdir(parents=True)
        (card / 'fw_version/mec_fw_version').write_text('0x24\n')
        for name, value in (('vendor', '0x1002'), ('device', '0x1586'), ('revision', '0xc1')):
            (card / name).write_text(value + '\n')
        for node, (target, simd) in enumerate((('0', '0'), (gfx, '80'))):
            path = root / f'kfd/{node}'
            path.mkdir(parents=True)
            (path / 'properties').write_text(f'simd_count {simd}\ngfx_target_version {target}\n'
                                             f'vendor_id 4098\ndevice_id 5510\n')
        (root / 'version').write_text('7.2.4\n')

    def identity(self, root):
        with patch.object(bc, 'DRM', root / 'drm'), patch.object(bc, 'KFD_NODES', root / 'kfd'), \
                patch.object(bc, 'ROCM_VERSION', root / 'version'), patch.object(bc, 'libraries', return_value={}):
            return bc.platform_identity(Path('/bin/true'), {}, None)

    def test_pci_ids_and_gfx_target_are_in_the_key(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.sysfs(Path(a))
            self.sysfs(Path(b), gfx='110500')
            first, other = self.identity(Path(a)), self.identity(Path(b))
        self.assertEqual(first['device']['pci'], {'card1/vendor': '0x1002', 'card1/device': '0x1586',
                                                  'card1/revision': '0xc1'})
        self.assertEqual(first['device']['gfx'], [{'gfx_target_version': '110501', 'vendor_id': '4098',
                                                   'device_id': '5510', 'simd_count': '80'}])
        self.assertNotEqual(bc.cache_key(inputs(platform=first)), bc.cache_key(inputs(platform=other)))

    def test_no_gpu_node_fails(self):
        with tempfile.TemporaryDirectory() as a:
            root = Path(a)
            self.sysfs(root, gfx='0')
            with self.assertRaisesRegex(ValueError, 'gfx target'):
                self.identity(root)


class FileMemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ckpt = self.root / 'ckpt'
        (self.ckpt / 'sub').mkdir(parents=True)
        (self.ckpt / 'config.json').write_text('{}')
        (self.ckpt / 'model-1.safetensors').write_bytes(b'w' * 1000)
        (self.ckpt / 'sub' / 'extra.bin').write_bytes(b'x')
        self.cache = self.root / 'cache'

    def identity(self):
        return bc.checkpoint_files(self.ckpt, bc.FileMemo(self.cache))

    def test_every_file_with_name_size_and_sha(self):
        files = self.identity()
        self.assertEqual([f['name'] for f in files], ['config.json', 'model-1.safetensors', 'sub/extra.bin'])
        self.assertEqual(files[1]['bytes'], 1000)
        self.assertEqual(files[1]['sha256'], g.digest(self.ckpt / 'model-1.safetensors'))

    def test_memo_hashes_once_and_rehashes_on_any_change(self):
        first = self.identity()
        with patch.object(bc, 'digest', side_effect=AssertionError('hashed again')):
            self.assertEqual(self.identity(), first)
        path = self.ckpt / 'model-1.safetensors'
        info = path.stat()
        # Same size and restored mtime: the inode change time still forces a full re-hash.
        time.sleep(0.01)
        path.write_bytes(b'v' * 1000)
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
        changed = self.identity()
        self.assertNotEqual(changed[1]['sha256'], first[1]['sha256'])
        self.assertEqual(changed[1]['sha256'], g.digest(path))


def fake_capture(directory, rows=2, payload=b'\x00\x00\x80\x3f', ids='7'):
    directory.mkdir(parents=True)
    dump = directory / 'dump'
    dump.mkdir()
    for name in g.NAMES:
        (dump / name).write_bytes(payload * g.WIDTH * rows)
    (directory / 'run.json').write_text('{"returncode": 0}\n')
    (directory / 'stdout.log').write_text(f'ids: {ids}\n')
    (directory / 'stderr.log').write_text('loaded\n')
    (directory / 'ids.txt').write_text(f'{ids}\n')
    record = {'returncode': 0, 'started_ns': 1, 'ended_ns': 2, 'queued_command': ['q'], 'output_ids': [int(ids)],
              'dumps': {name: {'sha256': g.digest(dump / name)} for name in g.NAMES}}
    return record


def store_worker(cache, key, source, queue):
    try:
        entry = bc.store(Path(cache), key, inputs(), Path(source), {'returncode': 0, 'dumps': {}},
                         {'host': 'test'}, check_dumps=False)
        queue.put(('ok', str(entry) if entry else None))
    except Exception as error:  # report every failure to the parent test
        queue.put(('error', repr(error)))


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.writable)
        self.root = Path(self.temp.name)
        self.cache = self.root / 'cache'
        self.inputs = inputs()
        self.key = bc.cache_key(self.inputs)
        self.record = fake_capture(self.root / 'capture')

    def writable(self):
        for path in self.root.rglob('*'):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o755)

    def store(self):
        return bc.store(self.cache, self.key, self.inputs, self.root / 'capture', self.record,
                        {'host': 'the test machine', 'started_ns': 1})

    def test_store_then_lookup_round_trip_read_only(self):
        entry = self.store()
        self.assertEqual(entry, self.cache / 'entries' / self.key)
        for path in [entry, *entry.rglob('*')]:
            self.assertEqual(stat.S_IMODE(path.stat().st_mode) & 0o222, 0, path)
        found, refused = bc.lookup(self.cache, self.key, self.inputs, rows=2)
        self.assertIsNone(refused)
        self.assertEqual(found['entry'], str(entry))
        self.assertEqual(found['output_ids'], [7])
        self.assertEqual(found['provenance']['host'], 'the test machine')
        self.assertEqual(len(found['manifest_sha256']), 64)
        self.assertEqual(found['dumps'][g.NAMES[0]], self.record['dumps'][g.NAMES[0]]['sha256'])
        self.assertEqual(list((self.cache / 'entries').iterdir()), [entry])

    def test_miss(self):
        self.assertEqual(bc.lookup(self.cache, self.key, self.inputs, rows=2), (None, None))

    def test_store_refuses_a_copy_that_differs_from_the_checked_capture(self):
        self.record['dumps'][g.NAMES[3]]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'differs from the checked capture'):
            self.store()
        self.assertEqual(list((self.cache / 'entries').iterdir()), [])

    def retired(self):
        return {p for p in (self.cache / 'entries').glob(f'.bad.{self.key}.*') if p.is_dir()}

    def corrupt_and_lookup(self, damage):
        before = self.retired() if self.cache.exists() else set()
        entry = self.store()
        self.writable()
        damage(entry)
        found, refused = bc.lookup(self.cache, self.key, self.inputs, rows=2)
        self.assertIsNone(found)
        self.assertIsNotNone(refused)
        # A refused entry leaves the key free for a fresh fill and stays for inspection.
        self.assertFalse(entry.exists())
        (new,) = self.retired() - before
        self.assertIn(refused, (new.parent / f'{new.name}.reason').read_text())
        return refused

    def test_corrupt_dump_byte_is_refused(self):
        def damage(entry):
            path = entry / 'dump' / g.NAMES[10]
            path.chmod(0o644)
            data = bytearray(path.read_bytes())
            data[100] ^= 1
            path.write_bytes(bytes(data))
        self.assertIn('h_10.f32', self.corrupt_and_lookup(damage))

    def test_missing_extra_and_manifest_damage_are_refused(self):
        cases = {
            'missing': lambda e: (e / 'dump' / g.NAMES[0]).unlink(),
            'extra': lambda e: (e / 'dump' / 'h_49.f32').write_bytes(b'x'),
            'manifest': lambda e: ((e / 'manifest.json').chmod(0o644), (e / 'manifest.json').write_text('{')),
            'symlink': lambda e: ((e / 'ids.txt').unlink(), (e / 'ids.txt').symlink_to(self.root / 'capture/ids.txt')),
            'key': lambda e: ((e / 'key.json').chmod(0o644), (e / 'key.json').write_text('{"inputs": {}}')),
        }
        for name, damage in cases.items():
            with self.subTest(name=name):
                self.corrupt_and_lookup(damage)
                self.writable()

    def test_rows_mismatch_is_refused(self):
        self.store()
        found, refused = bc.lookup(self.cache, self.key, self.inputs, rows=3)
        self.assertIsNone(found)
        self.assertIn('size', refused)

    def test_other_inputs_under_the_same_key_are_refused(self):
        self.store()
        found, refused = bc.lookup(self.cache, self.key, inputs(ids_sha256='0' * 64), rows=2)
        self.assertIsNone(found)
        self.assertIn('inputs', refused)

    def test_concurrent_fill_leaves_one_complete_entry(self):
        context = multiprocessing.get_context('fork')
        queue = context.Queue()
        workers = [context.Process(target=store_worker, args=(self.cache, self.key, self.root / 'capture', queue))
                   for _ in range(6)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(60)
        results = [queue.get(timeout=5) for _ in workers]
        self.assertTrue(all(kind == 'ok' for kind, _ in results), results)
        winners = [value for _, value in results if value]
        self.assertEqual(len(winners), 1, results)
        self.assertEqual(sorted(p.name for p in (self.cache / 'entries').iterdir()), [self.key])
        found, refused = bc.lookup(self.cache, self.key, self.inputs, rows=2)
        self.assertIsNone(refused)
        self.assertIsNotNone(found)


class RecheckScheduleTests(unittest.TestCase):
    def test_one_of_every_n_uses_and_never_for_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            due = [bc.record_use(cache, 'k' * 64, 3, {'output': str(n)})['due'] for n in range(7)]
            self.assertEqual(due, [True, False, False, True, False, False, True])
            self.assertEqual(bc.record_use(cache, 'k' * 64, 3, {})['index'], 7)
            self.assertFalse(any(bc.record_use(cache, 'j' * 64, 0, {})['due'] for _ in range(4)))


if __name__ == '__main__':
    unittest.main()
