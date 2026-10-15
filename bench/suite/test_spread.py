import contextlib
import io
import json
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

import spread

BASE = Path(__file__).parent/'results/2026-10-03-main-c286bda'


class SpreadTests(unittest.TestCase):
    def test_three_sessions_and_exact_ids(self):
        report = json.loads((BASE/'short/results.json').read_text())
        reports = [{**report, 'rows':[{**row, 'decode_tps':row['decode_tps']*factor} for row in report['rows']]}
                   for factor in (.98,1,1.02)]
        result = spread.summarize(reports, 3)
        self.assertEqual(len(result['rows']), 8)
        first = next(row for row in result['rows'] if row['name']==report['rows'][0]['name'] and row['spec']==0)
        self.assertEqual(first['samples'], 3)
        value = report['rows'][0]['decode_tps']
        self.assertAlmostEqual(first['decode_tps']['min'], value*.98)
        self.assertAlmostEqual(first['decode_tps']['median'], value)
        self.assertAlmostEqual(first['decode_tps']['max'], value*1.02)
        self.assertIn('min / median / max', spread.table(result))

    def test_incomplete_changed_or_mismatched_reports(self):
        report = json.loads((BASE/'short/results.json').read_text())
        for reports in ([report,report], [report,report,{**report,'config_sha256':'wrong'}],
                        [report,report,{**report,'rows':report['rows'][:-1]}],
                        [report,report,{**report,'rows':[{**row,'output_ids':[]} for row in report['rows']]}],
                        [report,report,{**report,'rows':[{**row,'decode_tps':float('nan')} for row in report['rows']]}]):
            with self.subTest(size=len(reports)), self.assertRaises(ValueError): spread.summarize(reports, 3)

    def test_cli_full_suite_and_rejected_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for repeat in range(3):
                for mode in ('short','prefill-4096','prefill-16384','prefill-65536'):
                    path = root/f'{repeat}-{mode}.json'
                    report = json.loads((BASE/mode/'results.json').read_text())
                    path.write_text(json.dumps({**report,'test_session':repeat}))
                    files.append(str(path))
            output = root/'spread.json'
            with patch('sys.argv',['spread.py',*files,'--expect-full-suite','--output',str(output)]), contextlib.redirect_stdout(io.StringIO()):
                spread.main()
            self.assertEqual(len(json.loads(output.read_text())['inputs']),12)
            self.assertTrue(output.with_suffix('.md').exists())
            for invalid in ([files[0],files[0]], files[:2]):
                with patch('sys.argv',['spread.py',*invalid,'--expect-full-suite','--output',str(output)]), \
                     contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): spread.main()

    def test_renamed_copies_are_not_three_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for repeat in range(3):
                for mode in ('short','prefill-4096','prefill-16384','prefill-65536'):
                    path = root/f'{repeat}-{mode}.json'
                    path.write_bytes((BASE/mode/'results.json').read_bytes())
                    files.append(str(path))
            output = root/'spread.json'
            with patch('sys.argv',['spread.py',*files,'--expect-full-suite','--output',str(output)]), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): spread.main()
            self.assertFalse(output.exists())
        original = json.loads((BASE/'short/results.json').read_text())
        altered = [{**original, 'test_session':repeat, 'queued_command':['/queue','same-run']} for repeat in range(3)]
        with self.assertRaises(ValueError): spread.summarize(altered,3)

    def test_output_alias_and_input_changes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = json.loads((BASE/'short/results.json').read_text())
            files = []
            for repeat in range(3):
                path = root/f'input-{repeat}.md'
                path.write_text(json.dumps({**original,'test_session':repeat}))
                files.append(path)
            before = files[0].read_bytes()
            for output in (files[0],files[0].with_suffix('.json')):
                with patch('sys.argv',['spread.py',*map(str,files),'--output',str(output)]), \
                     contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): spread.main()
                self.assertEqual(files[0].read_bytes(),before)
            summarize = spread.summarize
            def changed(reports,samples):
                result = summarize(reports,samples)
                files[0].write_text('{}')
                return result
            output = root/'new-summary.json'
            with patch.object(spread,'summarize',side_effect=changed), \
                 patch('sys.argv',['spread.py',*map(str,files),'--output',str(output)]), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): spread.main()
            self.assertFalse(output.exists())

    def test_metric_distribution_errors(self):
        for values in ([None,1], [float('inf')], [-1]):
            with self.assertRaises(ValueError): spread.distribution(values)
        with self.assertRaises(ValueError): spread.summarize([],3)
        with self.assertRaises(ValueError): spread.summarize([],1)

    def test_short_and_long_groups(self):
        reports = [{**json.loads((BASE/mode/'results.json').read_text()), 'test_session':repeat}
                   for repeat in range(3) for mode in ('short','prefill-4096','prefill-16384','prefill-65536')]
        result = spread.summarize(reports, 3)
        self.assertEqual(len(result['rows']), 11)
        self.assertIsNone(next(row for row in result['rows'] if row['name']=='long_4096')['decode_tps'])


if __name__ == '__main__': unittest.main()
