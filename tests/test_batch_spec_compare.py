from pathlib import Path
import shutil
import os
import struct
import json
from hashlib import sha256
import tempfile
import unittest
from bench.batch_spec_compare import compare

class SpecCompareTests(unittest.TestCase):
    def fixture(self, root):
        for request in range(2):
            (root / f'ids_{request}.i32').write_bytes(struct.pack('<2i', 1, 2))
            step = root / f'request_{request}' / 'step_0'
            step.mkdir(parents=True)
            for name in ('input.i32', 'history.i32'):
                (step / name).write_bytes(struct.pack('<i', 1))
            (step / 'target.i32').write_bytes(struct.pack('<i', 2))
            (step / 'logits.f32').write_bytes(bytes(248320 * 4))
            (step / 'stacks.f32').write_bytes(bytes(10240 * 4))
            (step / 'schedule.txt').write_text(f'request={request} generation=1 base=0 rows=1 replay=0 mpos=0\n')
            for index in range(161):
                (step / f'state_{index}.bin').write_bytes(b'\0')

    def identities(self, ref, actual):
        producers = {}
        for mode, root in [('native', ref), ('packed', actual)]:
            binary = root.parent / f'{mode}.elf'; binary.write_bytes(mode.encode())
            source = ('c' if mode == 'native' else 'd') * 64
            info = dict(binary_sha256=sha256(binary.read_bytes()).hexdigest(), source_manifest_sha256=source, binary_path=str(binary))
            producers[mode] = info
            run_mode = 'alone' if mode == 'native' else 'packed'
            argv = [str(binary), '/checkpoint', '/paths', str(root), run_mode, 'capture', '2', '3', '/blocks']
            data = dict(mode=run_mode, binary_sha256=info['binary_sha256'], source_sha256=source,
                        command_hex=('\0'.join(argv)+'\0').encode().hex(), cwd_hex=str(root.parent).encode().hex())
            (root.parent / (root.name+'_provenance.json')).write_text(json.dumps(data))
        return dict(schema='batch-spec-diagnostic-v1', producers=producers)

    def test_empty_and_same_folder_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); ref = root/'ref'; actual = root/'actual'
            ref.mkdir(); actual.mkdir()
            with self.assertRaises(ValueError): compare(ref, actual)
            self.fixture(ref)
            with self.assertRaises(ValueError): compare(ref, ref)

    def test_exact_and_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); ref = root/'ref'; ref.mkdir(); self.fixture(ref)
            actual = root/'actual'; shutil.copytree(ref, actual)
            record = self.identities(ref, actual)
            self.assertEqual(compare(ref, actual, record)['status'], 'PASS')
            (actual/'request_0/step_0/state_8.bin').write_bytes(b'\1')
            with self.assertRaises(ValueError): compare(ref, actual, record)

    def test_independent_identity_failures(self):
        cases = ('same_elf', 'changed_elf', 'wrong_root', 'wrong_argv0', 'settings', 'bench',
                 'role', 'source', 'symlink', 'extra_key')
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); ref = root/'ref'; ref.mkdir(); self.fixture(ref)
                actual = root/'actual'; shutil.copytree(ref, actual)
                record = self.identities(ref, actual)
                path = root/'actual_provenance.json'; data = json.loads(path.read_text())
                argv = bytes.fromhex(data['command_hex'])[:-1].decode().split('\0')
                if case == 'same_elf':
                    native = record['producers']['native']
                    record['producers']['packed'] = native
                    data = {**data, 'binary_sha256': native['binary_sha256'], 'source_sha256': native['source_manifest_sha256']}
                    argv[0] = native['binary_path']
                elif case == 'changed_elf': (root/'packed.elf').write_bytes(b'changed')
                elif case == 'wrong_root': argv[3] = str(ref)
                elif case == 'wrong_argv0': argv[0] = str(root/'native.elf')
                elif case == 'settings': argv[7] = '2'
                elif case == 'bench':
                    argv[5] = 'bench'
                    native_path = root/'ref_provenance.json'; native_data = json.loads(native_path.read_text())
                    native_argv = bytes.fromhex(native_data['command_hex'])[:-1].decode().split('\0')
                    native_argv[5] = 'bench'
                    native_path.write_text(json.dumps({**native_data, 'command_hex': ('\0'.join(native_argv)+'\0').encode().hex()}))
                elif case == 'role': data = {**data, 'mode': 'alone'}
                elif case == 'source': data = {**data, 'source_sha256': 'e'*64}
                elif case == 'extra_key': data = {**data, 'extra': 1}
                data = {**data, 'command_hex': ('\0'.join(argv)+'\0').encode().hex()}
                path.write_text(json.dumps(data))
                if case == 'symlink':
                    target = root/'linked.json'; path.rename(target); path.symlink_to(target)
                with self.assertRaises(ValueError): compare(ref, actual, record)

    def test_relative_request_paths_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); ref = root/'ref'; ref.mkdir(); self.fixture(ref)
            actual = root/'actual'; shutil.copytree(ref, actual)
            record = self.identities(ref, actual)
            path = root/'ref_provenance.json'; data = json.loads(path.read_text())
            argv = bytes.fromhex(data['command_hex'])[:-1].decode().split('\0')
            argv[1] = os.path.relpath('/checkpoint', root)
            path.write_text(json.dumps({**data, 'command_hex': ('\0'.join(argv)+'\0').encode().hex()}))
            self.assertEqual(compare(ref, actual, record)['status'], 'PASS')

    def test_copied_provenance_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); ref = root/'ref'; ref.mkdir(); self.fixture(ref)
            actual = root/'actual'; shutil.copytree(ref, actual)
            record = self.identities(ref, actual)
            shutil.copy2(root/'ref_provenance.json', root/'actual_provenance.json')
            with self.assertRaises(ValueError): compare(ref, actual, record)

    def test_missing_producer_identity_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); ref = root/'ref'; ref.mkdir(); self.fixture(ref)
            actual = root/'actual'; shutil.copytree(ref, actual)
            with self.assertRaises(ValueError): compare(ref, actual)

    def test_shared_missing_final_step_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); ref = root/'ref'; ref.mkdir(); self.fixture(ref)
            (ref/'ids_0.i32').write_bytes(struct.pack('<3i', 1, 2, 3))
            actual = root/'actual'; shutil.copytree(ref, actual)
            record = self.identities(ref, actual)
            with self.assertRaises(ValueError): compare(ref, actual, record)

    def test_shared_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); ref = root/'ref'; ref.mkdir(); self.fixture(ref)
            (ref/'request_0/step_0/logits.f32').unlink()
            actual = root/'actual'; shutil.copytree(ref, actual)
            record = self.identities(ref, actual)
            with self.assertRaises(ValueError): compare(ref, actual, record)

if __name__ == '__main__': unittest.main()
