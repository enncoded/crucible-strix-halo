import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import regression
import run as suite

BASE = suite.HERE / 'results/2026-10-03-main-c286bda'


# CRUCIBLE_PI_AI_DIR and CRUCIBLE_TEST_CKPT only enable optional server tests in the full `make test-cpu`
# environment. The runner refuses unknown CRUCIBLE_* names, so clear them here instead of widening the allow-list.
_TEST_DEPENDENCY_ENV = patch.dict(os.environ)


def setUpModule():
    _TEST_DEPENDENCY_ENV.start()
    for name in ("CRUCIBLE_PI_AI_DIR", "CRUCIBLE_TEST_CKPT"):
        os.environ.pop(name, None)


def tearDownModule():
    _TEST_DEPENDENCY_ENV.stop()


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline = regression.load_baseline(BASE)
        names = {'code_fibonacci', 'copy_python', 'long_4096'}
        self.rows = [copy.deepcopy(row) for row in self.baseline['rows'] if row['name'] in names]
        self.current = {**self.baseline, 'rows': self.rows}

    def test_equal_and_noise_boundary(self):
        verdict = regression.compare(self.current, self.baseline, 5)
        self.assertEqual(verdict['status'], 'PASS')
        slower = [{**row, 'decode_tps': row['decode_tps'] * 0.96} if row['decode_tps'] else row for row in self.rows]
        self.assertEqual(regression.compare({**self.current, 'rows': slower}, self.baseline, 5)['status'], 'PASS')
        slower = [{**row, 'decode_tps': row['decode_tps'] * 0.94} if row['decode_tps'] else row for row in self.rows]
        self.assertEqual(regression.compare({**self.current, 'rows': slower}, self.baseline, 5)['status'], 'FAIL')

    def test_prefill_and_ttft_regressions(self):
        for name, field, factor in (('long_4096', 'prefill_tps', .9), ('code_fibonacci', 'ttft_seconds', 1.1)):
            rows = [{**row, field: row[field] * factor} if row['name'] == name else row for row in self.rows]
            self.assertEqual(regression.compare({**self.current, 'rows': rows}, self.baseline, 5)['status'], 'FAIL')

    def test_invalid_evidence_fails(self):
        for mutation in ({'rows': []}, {'config_sha256': 'wrong'},
                         {'rows': self.rows + [self.rows[0]]},
                         {'rows': [{**row, 'output_ids': []} for row in self.rows]},
                         {'rows': [{**row, 'ttft_seconds': float('nan')} for row in self.rows]}):
            with self.subTest(mutation=list(mutation)), self.assertRaises(ValueError):
                regression.compare({**self.current, **mutation}, self.baseline, 5)
        for value in (-1, 51, float('nan'), float('inf')):
            with self.assertRaises(ValueError): regression.compare(self.current, self.baseline, value)

    def test_mixed_requests_keep_existing_schema(self):
        corpus = suite.load_corpus(suite.HERE/'manifest.json')
        prompts = [p for p in corpus['prompts'] if p['name'] in {'code_fibonacci','copy_python','long_4096'}]
        args = argparse.Namespace(tokens=160, chunk=8, spec=2, repeat=1, warmup=1,
                                  mode='regression', request_tokens={'long_4096': 1},
                                  prefill_names=('long_4096',), request_warmup={'long_4096': 0})
        path = self.root/'requests.txt'
        expected = suite.requests_file(path, prompts, args)
        self.assertEqual(len(expected), 5)
        lines = path.read_text().splitlines()
        self.assertEqual(len(lines), 9)
        self.assertEqual([line.split()[-4:] for line in lines if line.startswith('long_4096')], [['1','8','0','0']])

    def test_frozen_baseline_hashes(self):
        target = self.root/'baseline'
        target.mkdir()
        suite.save(target/'qualification.json', {'reports': {'short': 'wrong', 'prefill-4096': 'wrong'}})
        (target/'short').mkdir()
        (target/'short/results.json').write_text('{}')
        with self.assertRaises((ValueError, OSError)): regression.load_baseline(target)

    def test_launch_fake_queue_and_report(self):
        checkpoint = self.root/'checkpoint'
        checkpoint.mkdir()
        for name in ('config.json','quantization_config.json','tokenizer.json'): (checkpoint/name).write_text('{}')
        binary = self.root/'binary'
        binary.write_bytes(b'fake')
        suite.save(Path(str(binary)+'.build.json'), {'binary_sha256':suite.digest(binary), 'bulk_prefill':True,
                  'source_branch':'test','source_commit':'test'})
        corpus = suite.load_corpus(suite.HERE/'manifest.json')
        baseline = {**self.baseline, 'config_sha256':suite.digest(checkpoint/'config.json'),
                    'quantization_sha256':suite.digest(checkpoint/'quantization_config.json')}
        args = argparse.Namespace(checkpoint=checkpoint,binary=binary,baseline=BASE,output=self.root/'output',
                                  prompts='code_fibonacci,copy_python',noise_percent=5,gpu_run=Path('/queue'),gpu_wait=7200)
        def fake(command, stdout, stderr):
            self.assertEqual(command[:10],['/queue','-p','3','-w','7200','--','timeout','-k','10','300'])
            suite.save(Path(command[-2]), {'host':'fake','environment_policy_hashes':suite.policy_hashes()})
            stdout.write(json.dumps({'event':'ready','load_seconds':1,'draft_head':'full','draft_q':0,'lookup':0})+'\n')
            for line in Path(command[-4]).read_text().splitlines():
                name, _, n, _, spec, repeat = line.split()
                row = next(row for row in self.rows if row['name']==name and row['spec']==int(spec))
                stdout.write(json.dumps({**row,'event':'result','repeat':int(repeat)})+'\n')
            return subprocess.CompletedProcess(command,0)
        original_digest = suite.digest
        def digest(path):
            return corpus['tokenizer_sha256'] if path == checkpoint/'tokenizer.json' else original_digest(path)
        with patch.object(regression,'load_baseline',return_value=baseline), \
             patch.object(suite,'digest',side_effect=digest), patch.object(suite.subprocess,'run',side_effect=fake):
            regression.launch(args)
        self.assertEqual(json.loads((args.output/'regression.json').read_text())['status'],'PASS')
        self.assertEqual(len(json.loads((args.output/'a/results.json').read_text())['rows']),5)


if __name__ == '__main__': unittest.main()
