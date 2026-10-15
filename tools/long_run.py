"""Additional long checks. Primary Gates 1, 2, and 3 remain mandatory."""
from pathlib import Path

import gates as g
import long_capture as capture
import long_gates as lg


def persist(out,plan):
    g.save(out/'long-report.json',plan)


def verify(plan):
    if g.digest(g.ROOT/'bench/suite/kl/manifest.json') != plan['fixed_corpus_sha256']:
        raise ValueError('fixed KL corpus manifest changed during captures')
    for role in ('candidate','baseline'):
        if g.digest(Path(plan[role])) != plan[role+'_sha256']:
            raise ValueError('long engine binary changed during captures')
        probe = plan[role+'_probe']
        if probe and capture.probe_identity(Path(probe['path']),Path(plan[role])) != probe:
            raise ValueError('long diagnostic probe changed during captures')
    if any(g.digest(Path(item['file'])) != item['sha256'] for item in plan['prompts']):
        raise ValueError('long prompt changed during captures')
    if any(g.digest(g.ROOT/'tools'/name) != value for name,value in plan['tool_hashes'].items()):
        raise ValueError('long capture tool changed during captures')
    if plan.get('resident') is not None:
        import resident_capture
        resident_capture.verify(plan,plan['resident'])
    for record in plan['captures'].values():
        if g.digest(Path(record['logit_path'])) != record['logits']['sha256']:
            raise ValueError('long primary logits changed after capture')
        if g.digest(Path(record['stdout_path'])) != record['stdout_sha256']:
            raise ValueError('long primary stdout changed after capture')
        if record.get('positions_path') and g.digest(Path(record['positions_path'])) != record['positions_sha256']:
            raise ValueError('long fixed-position file changed after capture')


def same_bytes(left,right):
    with Path(left).open('rb') as a,Path(right).open('rb') as b:
        while True:
            av,bv = a.read(1<<20),b.read(1<<20)
            if av != bv: return False
            if not av: return True


def equality(left,right):
    if (not same_bytes(left['logit_path'],right['logit_path']) or
            left['output_ids'] != right['output_ids']):
        raise ValueError('long production candidate/baseline full logits or complete IDs differ')
    return {'full_final_vocabulary_logits':'bitwise equal','complete_output_ids':'equal'}


def point(prompt,position,phase,candidate,baseline,control):
    return {'prompt':prompt,'position':position,'phase':phase,'production':phase=='production schedule',
            'candidate':lg.divergence(candidate,control),'baseline':lg.divergence(baseline,control),
            'control_top1':max(range(lg.VOCAB),key=control.__getitem__)}


def diagnostics(plan,prompt,records):
    native_candidate = next(lg.rows(records['candidate']['logit_path'],1))
    native_baseline = next(lg.rows(records['baseline']['logit_path'],1))
    native_control = next(lg.rows(records['control']['logit_path'],1))
    final = point(prompt['file'],prompt['tokens']-1,'production schedule',native_candidate,native_baseline,native_control)
    names = ('candidate-clipped','baseline-clipped','control-clipped')
    iterators = [lg.rows(records[name]['logit_path'],records[name]['point_count']) for name in names]
    points = records[names[0]]['points']
    if (records[names[1]]['points'] != points or
            records[names[1]]['schedule'] != records[names[0]]['schedule']):
        raise ValueError('candidate and baseline clipped schedules differ')
    if records[names[2]]['points'] != points:
        raise ValueError('eight-row control sampled points differ')
    samples,last = [],None
    for position,values in zip(points,zip(*iterators)):
        samples.append(point(prompt['file'],position,'clipped schedule',*values))
        last = values
    invariant = {'candidate':last[0].tobytes()==native_candidate.tobytes(),
                 'baseline':last[1].tobytes()==native_baseline.tobytes(),
                 'control':last[2].tobytes()==native_control.tobytes(),'scope':'full final-position vocabulary logits only',
                 'warning':'clipped samples do not establish all-position production numerics'}
    return samples,final,invariant


def run(args,plan,out,candidate_policy):
    out.mkdir()
    persist(out,plan)
    samples,finals = [],[]
    try:
        grouped=None
        if plan.get('resident') is not None:
            import resident_capture
            grouped=resident_capture.collect(args,plan,out,plan['resident'])
        for index,prompt in enumerate(plan['prompts']):
            records = {} if grouped is None else grouped[index]
            for role in ('candidate','baseline'):
                policy = candidate_policy if role=='candidate' else {
                    'native':plan['usage']['baseline']['native_draft_head'],
                    'mtp_prefix_capable':plan['usage']['baseline'].get('mtp_prefix_capable',False)}
                record = (capture.native(Path(plan[role]),prompt,out/f'{index}-{role}',plan['prefill_rows'],args,policy)
                          if grouped is None else records[role])
                records[role] = record
                plan['captures'][f'{index}-{role}'] = record
                persist(out,plan)
            if prompt['file'] in plan['bitwise_files']:
                plan.setdefault('bitwise',{})[prompt['file']] = equality(records['candidate'],records['baseline'])
            if prompt['file'] not in plan['kl_files']: continue
            control_policy = {'native':False,
                              'mtp_prefix_capable':plan['usage']['baseline'].get('mtp_prefix_capable',False)}
            record = (capture.native(Path(plan['baseline']),prompt,out/f'{index}-control',0,args,control_policy)
                      if grouped is None else records['control'])
            records['control'] = record
            plan['captures'][f'{index}-control'] = record
            for role,probe,rows in (('candidate',plan['candidate_probe'],plan['prefill_rows']),
                                   ('baseline',plan['baseline_probe'],plan['prefill_rows']),('control',plan['baseline_probe'],0)):
                record = (capture.clipped(probe,prompt,out/f'{index}-{role}-clipped',rows,args,reference=role!='candidate')
                          if grouped is None else records[role+'-clipped'])
                records[role+'-clipped'] = record
                plan['captures'][f'{index}-{role}-clipped'] = record
                persist(out,plan)
            current,last,invariant = diagnostics(plan,prompt,records)
            samples.extend(current);finals.append(last)
            plan.setdefault('boundary_invariance',{})[prompt['file']] = invariant
        if samples:
            plan['sampled_kl'] = lg.assess(samples,plan['absolute_kl_max'])
            plan['production_final_kl'] = lg.assess(finals,plan['absolute_kl_max'])
            if any(plan[key]['status'] != 'PASS' for key in ('sampled_kl','production_final_kl')):
                raise ValueError('long sampled or production median KL/top-one policy failed')
        verify(plan)
        plan['status'] = 'PASS'
        persist(out,plan)
    except (ValueError,OSError) as error:
        plan['status'],plan['error'] = 'FAIL',str(error)
        persist(out,plan)
        raise
    return plan
