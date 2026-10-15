"""CPU tests for the CRUCIBLE_MTP_SIDECAR provenance in the gate reports."""
import hashlib
import os
from pathlib import Path
import sys
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import batch_gate_report as b  # noqa: E402
import gates as g  # noqa: E402

KNOB = 'CRUCIBLE_MTP_SIDECAR'


class SidecarTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.file = Path(self.dir.name) / 'side.safetensors'
        self.file.write_bytes(b'sidecar bytes')

    def tearDown(self):
        self.dir.cleanup()

    def test_knob_is_published(self):
        self.assertIn(KNOB, g.RUNTIME_KNOBS)
        self.assertIn(KNOB, b.RUNTIME_KNOBS)
        for module in (g, b):
            published = module.runtime_environment({KNOB: str(self.file)})
            self.assertEqual(published[KNOB], 'set (see mtp_sidecar)')
            self.assertNotIn(self.dir.name, repr(published))
            self.assertIsNone(module.runtime_environment({})[KNOB])

    def test_cache_key_does_not_publish_the_path(self):
        self.assertNotIn(KNOB, g.CACHE_PUBLIC_KNOBS)
        import baseline_cache as bc
        key_env = bc.output_environment({KNOB: str(self.file)}, g.CACHE_PUBLIC_KNOBS)
        self.assertNotIn(self.dir.name, repr(key_env))

    def test_reference_capture_keeps_the_variable(self):
        # Gate 1 baseline captures score no drafts, so the sidecar cannot change them. Stripping it would change
        # the pinned capture sources and invalidate every cached baseline (baseline_cache.CAPTURE_SOURCE_SHA256).
        with patch.dict(g.os.environ, {KNOB: str(self.file)}):
            self.assertEqual(g.capture_environment(reference=True)[KNOB], str(self.file))

    def test_record_has_hash_and_size(self):
        for module in (g, b):
            rec = module.mtp_sidecar_record({KNOB: str(self.file)})
            self.assertEqual(rec, {'bytes': 13, 'sha256': hashlib.sha256(b'sidecar bytes').hexdigest()})

    def test_unset_or_empty_records_none(self):
        for module in (g, b):
            self.assertIsNone(module.mtp_sidecar_record({}))
            self.assertIsNone(module.mtp_sidecar_record({KNOB: ''}))

    def test_refuses_what_is_not_a_regular_file(self):
        link = Path(self.dir.name) / 'link.safetensors'
        os.symlink(self.file, link)
        for module in (g, b):
            for bad in (link, Path(self.dir.name), Path(self.dir.name) / 'missing'):
                with self.assertRaisesRegex(ValueError, 'regular file') as caught:
                    module.mtp_sidecar_record({KNOB: str(bad)})
                # The message reaches report.json; the path names the user's home.
                self.assertNotIn(self.dir.name, str(caught.exception))


class PinTests(unittest.TestCase):
    """The GPU wait can last 21,600 s: the gate must run and record a private copy, not the source file."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.file = self.root / 'side.safetensors'
        self.file.write_bytes(b'sidecar bytes')
        self.record = {'bytes': 13, 'sha256': hashlib.sha256(b'sidecar bytes').hexdigest()}

    def test_pin_copies_read_only_and_points_the_variable_at_the_copy(self):
        with patch.dict(g.os.environ, {KNOB: str(self.file)}):
            self.assertEqual(g.pin_sidecar(self.root / 'bin'), self.record)
            copy = Path(g.os.environ[KNOB])
            self.assertEqual(copy.parent, (self.root / 'bin').resolve())
            self.assertEqual(copy.stat().st_mode & 0o777, 0o444)
            # the source is replaced after the pin; captures and records keep the pinned bytes.
            self.file.write_bytes(b'other sidecar bytes')
            job = g.prepare_capture(Path('/engine'), Path('/ckpt'), Path('/ids'), self.root / 'cap', 4, 2, 8, 0, 0,
                                    Path('/gpu-run'))
            self.assertEqual(job['env'][KNOB], str(copy))
            self.assertEqual(job['record']['mtp_sidecar'], self.record)
            self.assertEqual(json.loads((self.root / 'cap/run.json').read_text())['mtp_sidecar'], self.record)

    def test_pin_without_the_variable_does_nothing(self):
        with patch.dict(g.os.environ, {KNOB: ''}):
            self.assertIsNone(g.pin_sidecar(self.root / 'bin'))
            self.assertFalse((self.root / 'bin').exists())

    def test_cache_key_follows_the_bytes_not_the_pinned_path(self):
        import baseline_cache as bc
        keys = []
        for name in ('a', 'b'):
            with patch.dict(g.os.environ, {KNOB: str(self.file)}):
                g.pin_sidecar(self.root / name)
                keys.append(g.cache_environment(g.capture_environment(reference=True)))
        self.assertEqual(keys[0][KNOB], keys[1][KNOB])
        self.assertNotIn(self.dir.name, repr(bc.output_environment(keys[0], g.CACHE_PUBLIC_KNOBS)))
        with patch.dict(g.os.environ, {KNOB: str(self.file)}):
            self.file.write_bytes(b'changed')
            self.assertNotEqual(g.cache_environment(g.capture_environment())[KNOB], keys[0][KNOB])

    def test_baseline_cache_plan_keys_on_the_bytes(self):
        import baseline_cache as bc
        binary, ids = self.root / 'baseline-engine', self.root / 'ids'
        binary.write_bytes(b'engine')
        ids.write_text('1,2\n')
        args = SimpleNamespace(baseline_cache=self.root / 'cache', baseline_build_record=None)
        keys = []
        with patch.object(bc, 'checkpoint_files', return_value={}), \
                patch.object(bc, 'platform_identity', return_value={}):
            for name in ('a', 'b'):
                with patch.dict(g.os.environ, {KNOB: str(self.file)}):
                    g.pin_sidecar(self.root / name)
                    keys.append(g.baseline_cache_plan(args, binary, self.root, ids, 2, 0,
                                                      {'mtp_prefix_capable': False})['key'])
        self.assertEqual(keys[0], keys[1])

    def gate_args(self):
        engine = self.root / 'engine'
        engine.write_bytes(b'engine')
        return SimpleNamespace(output=self.root / 'gate', engine=engine, engine_build_record=None,
                               checkpoint=self.root / 'missing-checkpoint', ids=self.root / 'ids')

    def test_gate_refuses_a_bad_sidecar_before_any_subprocess(self):
        link = self.root / 'link.safetensors'
        os.symlink(self.file, link)
        args = self.gate_args()
        with patch.dict(g.os.environ, {KNOB: str(link)}), patch.object(g, 'disk_preflight', return_value=0), \
                patch.object(g.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'regular file'):
                g.run(args)
            self.assertEqual(run.call_count, 0)
            self.assertEqual(g.os.environ[KNOB], str(link))
        self.assertIn('regular file', json.loads((args.output / 'report.json').read_text())['error'])

    def test_gate_report_records_the_pin_and_restores_the_variable(self):
        args = self.gate_args()
        with patch.dict(g.os.environ, {KNOB: str(self.file)}), patch.object(g, 'disk_preflight', return_value=0), \
                patch.object(g.subprocess, 'run') as run:
            with self.assertRaises(OSError):
                g.run(args)  # the checkpoint is missing: the gate stops after the pins
            self.assertEqual(run.call_count, 0)
            self.assertEqual(g.os.environ[KNOB], str(self.file))
        report = json.loads((args.output / 'report.json').read_text())
        self.assertEqual(report['mtp_sidecar'], self.record)
        self.assertEqual((args.output / 'bin/mtp_sidecar.safetensors').read_bytes(), b'sidecar bytes')
        self.assertNotIn(self.dir.name + '/side', json.dumps(report))


class EngineLineTests(unittest.TestCase):
    """An engine that ignores the variable runs the stock head: the gate must see the engine's own line."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.copy = self.root / 'mtp_sidecar.safetensors'
        self.copy.write_bytes(b'sidecar bytes')
        self.record = {'bytes': 13, 'sha256': hashlib.sha256(b'sidecar bytes').hexdigest()}

    def job(self, stderr, reference=False, record=True):
        (self.root / 'stderr.log').write_text(stderr)
        return {'output': self.root, 'env': {KNOB: str(self.copy)}, 'reference': reference,
                'record': {'mtp_sidecar': self.record if record else None, 'started_ns': 0}}

    def test_line_with_the_recorded_size_passes(self):
        g.check_sidecar_log(self.job('x\nmtp sidecar: 17 tensors from mtp_sidecar.safetensors (13 bytes)\ny\n'))

    def test_missing_wrong_size_or_old_format_fails(self):
        for text in ('', 'mtp sidecar: 17 tensors from mtp_sidecar.safetensors (14 bytes)\n',
                     'mtp sidecar: 0 tensors from mtp_sidecar.safetensors (13 bytes)\n',
                     f'mtp sidecar: 17 tensors from {self.copy}\n'):
            with self.assertRaisesRegex(ValueError, 'mtp sidecar'):
                g.check_sidecar_log(self.job(text))

    def test_reference_or_unset_needs_no_line(self):
        g.check_sidecar_log(self.job('', reference=True))
        g.check_sidecar_log(self.job('', record=False))

    def test_a_changed_copy_fails(self):
        job = self.job('mtp sidecar: 17 tensors from mtp_sidecar.safetensors (13 bytes)\n')
        self.copy.write_bytes(b'sidecar bytez')
        with self.assertRaisesRegex(ValueError, 'changed'):
            g.check_sidecar_log(job)

    def test_finish_capture_checks_the_line_first(self):
        job = {**self.job(''), 'rows': 1, 'tokens': 1}
        with self.assertRaisesRegex(ValueError, 'mtp sidecar'):
            g.finish_capture(job, 0, 1)


if __name__ == '__main__':
    unittest.main()
