"""CPU tests for fresh long captures and the shared KL policy."""
import array
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import long_gates as lg
import gates as g
import build_long_probe as builder
from types import SimpleNamespace


class LongGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_fixed_points_and_schedule(self):
        points = lg.sample_points(1024)
        self.assertEqual(points, tuple(sorted(set(range(255,1024,256)) | set(range(960,1024)))))
        schedule = lg.schedule(1024,points,4096)
        self.assertEqual(schedule[:4], ((0,256),(256,256),(512,256),(768,193)))
        self.assertEqual(schedule[-1],(1023,1))
        self.assertEqual(sum(count for _,count in schedule),1024)
        for points in ((), (255,), (511,255,1023), (255,255,1023)):
            with self.assertRaises(ValueError): lg.schedule(1024,points,4096)

    def test_kl_stable_both_directions(self):
        equal = lg.divergence([0.,1.,2.],[0.,1.,2.])
        self.assertEqual(equal['forward'],0)
        self.assertEqual(equal['reverse'],0)
        self.assertEqual(equal['top1'],2)
        changed = lg.divergence([0.,1.,2.],[1.,0.,2.])
        self.assertGreater(changed['forward'],0)
        self.assertGreater(changed['reverse'],0)
        large = lg.divergence([10000.,10001.],[10000.,10001.])
        self.assertEqual(large['forward'],0)
        for bad in ([0.,float('nan')],[float('inf'),0.]):
            with self.assertRaises(ValueError): lg.divergence(bad,[0.,1.])
        with self.assertRaises(ValueError): lg.divergence([0.],[0.,1.])

    def test_kl_matches_independent_decimal_reference(self):
        from decimal import Decimal,localcontext
        vectors=([0.,1.,2.],[0.,1.000001,2.],[10000.,10000.00001,9998.])
        with localcontext() as context:
            context.prec=70
            def log_probs(values):
                shifted=[Decimal.from_float(value)-Decimal.from_float(max(values)) for value in values]
                denominator=sum(value.exp() for value in shifted).ln()
                return [value-denominator for value in shifted]
            for left,right in ((vectors[0],vectors[1]),(vectors[1],vectors[2]),(vectors[2],vectors[2])):
                a,b=log_probs(left),log_probs(right)
                expected=sum(x.exp()*(x-y) for x,y in zip(a,b))
                actual=lg.divergence(left,right)['forward']
                self.assertAlmostEqual(actual,float(expected),delta=2e-15)

    def point(self, candidate=.001, baseline=.001, mismatch=False, margin=1.):
        return {'prompt':'p','position':1023,'production':True,
                'candidate':{'forward':candidate,'reverse':candidate,'top1':2},
                'baseline':{'forward':baseline,'reverse':baseline,'top1':2,'margin':margin},
                'control_top1':1 if mismatch else 2}

    def test_aggregate_not_individual_bound(self):
        points = [self.point(.1),self.point(0),self.point(0)]
        self.assertEqual(lg.assess(points,None)['status'],'PASS')
        self.assertEqual(lg.assess([self.point(.00301)],None)['status'],'FAIL')
        self.assertEqual(lg.assess([self.point(3e-5,0)],None)['status'],'PASS')
        self.assertEqual(lg.assess([self.point(3.01e-5,0)],None)['status'],'FAIL')
        self.assertEqual(lg.assess([self.point()],.0005)['status'],'FAIL')
        changed=self.point()
        changed['candidate']['top1']=1
        self.assertEqual(lg.assess([changed],None)['status'],'FAIL')
        changed['baseline']['margin']=.0005
        tie = lg.assess([changed],None)
        self.assertEqual(tie['status'],'PASS')
        self.assertEqual(len(tie['near_ties']),1)
        for values in ([],[self.point(float('nan'))]):
            with self.assertRaises(ValueError): lg.assess(values,None)
        for maximum in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): lg.assess([self.point()],maximum)

    def test_equal_prompt_weight_and_all_sample_top1(self):
        points = [dict(self.point(0),prompt='long') for _ in range(100)]
        points += [dict(self.point(.01),prompt=key) for key in ('short-a','short-b')]
        result = lg.assess(points,None)
        self.assertEqual(result['status'],'FAIL')
        self.assertEqual(result['medians']['candidate']['forward'],.01)
        changed = dict(self.point(mismatch=True),production=False)
        self.assertEqual(lg.assess([changed],None)['status'],'PASS')
        changed['candidate']['top1']=1
        self.assertEqual(lg.assess([changed],None)['status'],'FAIL')

    def build_args(self):
        source = self.root/'source'
        for name in ('engine/engine.hip','engine/cli/gen_cli.h','kernels/test.h','engine/exl3/safetensors.h'):
            file = source/name
            file.parent.mkdir(parents=True,exist_ok=True)
            file.write_text(name)
        engine = self.root/'engine'
        engine.write_bytes(b'engine')
        record = self.root/'engine.build.json'
        g.save(record,{'binary_sha256':g.digest(engine),'source_hashes':builder.source_hashes(source),
                       'command':['hipcc','-O3','-DMODEL_TEST=1','-ffast-math',str(source/'engine/engine.hip')]})
        return SimpleNamespace(source=source,engine=engine,engine_build_record=record,
                               output=self.root/'probe',hipcc='/compiler')

    def test_probe_build_binding_and_changed_record(self):
        args = self.build_args()
        def compiler(command,**kwargs):
            self.assertIn('-DMODEL_TEST=1',command)
            self.assertIn('-ffast-math',command)
            Path(command[-1]).write_bytes(b'compiled')
            return SimpleNamespace(returncode=0)
        with patch.object(builder.subprocess,'run',side_effect=compiler): builder.build(args)
        record = json.loads(Path(str(args.output)+'.build.json').read_text())
        self.assertEqual(record['bound_engine_binary_sha256'],g.digest(args.engine))
        self.assertEqual(record['capture_origin'],lg.PROBE_ORIGIN)
        args.output = self.root/'probe-changed'
        def changed(command,**kwargs):
            done = compiler(command,**kwargs)
            args.engine_build_record.write_text('{}')
            return done
        with patch.object(builder.subprocess,'run',side_effect=changed), self.assertRaises(ValueError): builder.build(args)
        self.assertFalse(Path(str(args.output)+'.build.json').exists())

    def test_probe_build_requires_one_selected_source(self):
        args=self.build_args()
        record=json.loads(args.engine_build_record.read_text())
        for sources in ([],['/tmp/untracked/engine/engine.hip'],[str(args.source/'engine/engine.hip')]*2):
            g.save(args.engine_build_record,{**record,'command':['hipcc','-O3',*sources]})
            def compiler(command,**kwargs):
                Path(command[-1]).write_bytes(b'compiled')
            with patch.object(builder.subprocess,'run',side_effect=compiler) as process,self.assertRaises(ValueError):
                builder.build(args)
            process.assert_not_called()
        g.save(args.engine_build_record,{**record,'command':['hipcc','-O3','engine/engine.hip'],
                                        'build_cwd':str(args.source)})
        with patch.object(builder.subprocess,'run',side_effect=compiler): builder.build(args)

    def test_resident_probe_uses_same_source_binding_and_numeric_flags(self):
        import build_resident_probe as resident
        args=self.build_args()
        def compiler(command,**kwargs):
            self.assertIn(str(g.ROOT/'bench/resident_probe.hip'),command)
            Path(command[-1]).write_bytes(b'resident-compiled')
        with patch.object(builder.subprocess,'run',side_effect=compiler): resident.build(args)
        record=json.loads(Path(str(args.output)+'.build.json').read_text())
        self.assertEqual(record['capture_origin'],resident.RESIDENT_ORIGIN)
        self.assertEqual(set(record['probe_hashes']),set(resident.RESIDENT_FILES))
        self.assertEqual(record['bound_engine_binary_sha256'],g.digest(args.engine))
        self.assertEqual(record['inherited_compiler_flags'],['-O3','-DMODEL_TEST=1','-ffast-math'])

    def test_probe_rejects_unrecorded_optimization_default(self):
        args=self.build_args()
        record=json.loads(args.engine_build_record.read_text())
        source=args.source/'engine/engine.hip'
        command=['hipcc','-std=c++20','--offload-arch=gfx1151',str(source)]
        with self.assertRaisesRegex(ValueError,'explicit optimization'):
            builder.compiler_flags(command,source)
        g.save(args.engine_build_record,{**record,'command':command})
        with patch.object(builder.subprocess,'run') as compiler:
            with self.assertRaisesRegex(ValueError,'explicit optimization'): builder.build(args)
        compiler.assert_not_called()

    def test_probe_compiler_flags_fail_closed(self):
        self.assertEqual(builder.compiler_flags('hipcc -O2 -D TEST=1 -U OLD -ffast-math'),
                         ('-O2','-DTEST=1','-UOLD','-ffast-math'))
        self.assertEqual(builder.compiler_flags('hipcc -O3 engine/engine.hip -o build/engine/main -lpthread',
                                                     g.ROOT/'engine/engine.hip',g.ROOT),('-O3',))
        for command in (None,[1],['-D'],['-include','extra.h'],['-include/tmp/header.h'],
                        ['-imacros/tmp/header.h'],['@response'],['-Xclang','hidden'],
                        ['-mllvm','-enable-unsafe-fp-math'],['-fplugin=/tmp/plugin.so'],
                        ['-I/tmp/external'],['--offload-arch=gfx1100'],['-std=c++17'],
                        ['hipcc','-O3','/tmp/untracked/engine/engine.hip']):
            with self.assertRaises(ValueError): builder.compiler_flags(command)
        args=self.build_args()
        args.output.write_bytes(b'old binary')
        with self.assertRaisesRegex(ValueError,'new probe'): builder.build(args)
        args.output=self.root/'missing-output'
        with patch.object(builder.subprocess,'run',return_value=SimpleNamespace(returncode=0)),self.assertRaisesRegex(ValueError,'did not create'):
            builder.build(args)
        args.engine.write_bytes(b'changed binary')
        with self.assertRaisesRegex(ValueError,'source and binary'): builder.build(args)

    def test_canonical_logits_and_freshness(self):
        path = self.root/'logits.f32'
        start = time.time_ns()
        time.sleep(.003)
        path.write_bytes(array.array('f',[0,1,2,3]).tobytes())
        with patch.object(lg,'VOCAB',4):
            self.assertEqual(lg.check_logits(path,1,start)['bytes'],16)
            with self.assertRaises(ValueError): lg.check_logits(path,2,start)
            path.write_bytes(array.array('f',[0,1,float('nan'),3]).tobytes())
            with self.assertRaises(ValueError): list(lg.rows(path,1))
            path.write_bytes(b'')
            with self.assertRaises(ValueError): lg.check_logits(path,1,start)
            path.unlink()
            with self.assertRaises(ValueError): lg.check_logits(path,1,start)


if __name__ == '__main__': unittest.main()
