"""Bounded long captures: one model load and one GPU grant for each role."""
import json
from pathlib import Path
from types import SimpleNamespace

import gates as g
import long_capture as capture
import long_gates as lg
from build_resident_probe import RESIDENT_FILES,RESIDENT_ORIGIN

MAX_JOBS=7
MAX_PROMPT_TOKENS=32768
MAX_OUTPUT_TOKENS=160


def identity(probe,engine):
    return capture.probe_identity(probe,engine,RESIDENT_FILES,RESIDENT_ORIGIN)


def groups(plan,mode):
    result=[]
    for role in ('candidate','baseline','control'):
        indices=[index for index,prompt in enumerate(plan['prompts'])
                 if role!='control' or prompt['file'] in plan['kl_files']]
        sharded=mode=='all' or (mode=='control' and role=='control')
        batches=[[index] for index in indices] if sharded else ([indices] if indices else [])
        result.extend({'role':role,'name':f'{role}-{batch[0]}' if sharded else role,'indices':batch}
                      for batch in batches)
    return result


def prepare(args,plan):
    mode=getattr(args,'long_resident_shard','none')
    budget=getattr(args,'long_resident_16k_control_timeout',300)
    if mode not in ('none','control','all') or type(budget) is not int or budget not in (300,600):
        raise ValueError('invalid resident shard or control timeout')
    if budget==600 and (mode=='none' or not any(prompt['tokens']==16384 and prompt['file'] in plan['kl_files']
                                             for prompt in plan['prompts'])):
        raise ValueError('the 600-second exception needs a sharded 16K KL control')
    files=tuple(getattr(args,name,None) for name in
                ('long_resident_candidate_probe','long_resident_baseline_probe'))
    if all(value is None for value in files):
        if mode!='none' or budget!=300: raise ValueError('resident sharding needs both source-bound probes')
        return None
    if any(value is None for value in files): raise ValueError('resident capture needs both source-bound probes')
    if (not 1<=len(plan['prompts'])<=MAX_JOBS or
            sum(prompt['tokens'] for prompt in plan['prompts'])>MAX_PROMPT_TOKENS):
        raise ValueError('resident role exceeds seven requests or 32768 prompt tokens')
    if not 1<=args.long_tokens<=MAX_OUTPUT_TOKENS:
        raise ValueError('resident capture needs 1 through 160 output IDs per request')
    if getattr(args,'mtp_prefix',None) not in (None,0):
        raise ValueError('resident long capture uses the explicit zero MTP prefix policy')
    probes={role:identity(Path(file),Path(plan[role])) for role,file in zip(('candidate','baseline'),files)}
    names=('resident_capture.py','build_resident_probe.py')
    selected=groups(plan,mode)
    return {'probes':probes,'origin':RESIDENT_ORIGIN,'grants':len(selected),'groups':selected,
            'shard_mode':mode,'control_16k_timeout_seconds':budget,
            'tool_hashes':{name:g.digest(g.ROOT/'tools'/name) for name in names}}


def safe_field(value):
    text=str(value)
    if not text or any(char in text for char in '\t\r\n\0'):
        raise ValueError('resident manifest fields cannot contain control characters')
    return text


def manifest(plan,out,role,tokens,indices=None,name=None):
    lines=[]
    inputs={}
    for index,prompt in enumerate(plan['prompts']):
        if indices is not None and index not in indices:continue
        if role=='control' and prompt['file'] not in plan['kl_files']:continue
        point_file=out/f'{index}-{role}-positions.ids'
        if prompt['file'] in plan['kl_files']:
            point_file.write_text(','.join(map(str,lg.sample_points(prompt['tokens'])))+'\n')
            points=str(point_file.resolve())
            inputs[points]=g.digest(point_file)
        else: points='-'
        fields=(prompt['file'],points,str((out/f'{name or role}-resident/{index}').resolve()),tokens)
        lines.append('\t'.join(safe_field(value) for value in fields))
        inputs[prompt['file']]=g.digest(Path(prompt['file']))
    path=out/f'{name or role}-manifest.tsv'
    path.write_text('\n'.join(lines)+'\n')
    return path,{**inputs,str(path):g.digest(path)},len(lines)


def check_tree(path,names):
    if path.is_symlink() or not path.is_dir() or {child.name for child in path.iterdir()}!=set(names):
        raise ValueError(f'unexpected resident artifact set: {path}')


def evidence(path,record,prompt,rows,tokens,points):
    check_tree(path,('stdout.log','logits.f32'))
    stdout=path/'stdout.log'
    g.regular(stdout,record['started_ns'])
    common={**record,'stdout_path':str(stdout),'stdout_sha256':g.digest(stdout),
            'resident_source':record['stdout_path'],
            'draft_policy':{'head':'full','head_source':'explicit resident options',
                            'adapt':0,'q':0,'margin':0,'lookup':0,'overlap':0,'mtp_prefix':0}}
    if points is None:
        return {**common,**capture.native_evidence(path/'logits.f32',stdout,record['started_ns'],tokens)}
    measured=capture.clipped_evidence(path/'logits.f32',stdout,record['started_ns'],prompt,rows)
    return {**common,**measured,'positions_path':str(points),'positions_sha256':g.digest(points)}


def collect_role(args,plan,out,prepared,role,indices=None,name=None):
    probe=prepared['probes']['candidate' if role=='candidate' else 'baseline']
    rows=0 if role=='control' else plan['prefill_rows']
    context=max(32768,max(prompt['tokens']+args.long_tokens for prompt in plan['prompts']))
    path,inputs,count=manifest(plan,out,role,args.long_tokens,indices,name)
    command=[probe['path'],str(args.checkpoint.resolve()),str(context),str(rows),str(path.resolve())]
    target=out/f'{name or role}-resident'
    exception=(capture.control_exception() if role=='control' and indices is not None and len(indices)==1
               and plan['prompts'][indices[0]]['tokens']==16384
               and prepared['control_16k_timeout_seconds']==600 else None)
    execution_args=SimpleNamespace(**{**vars(args),'priority':2}) if exception else args
    record=capture.execute(command,target,execution_args,timeout_exception=exception,reference=role!='candidate')
    events=[json.loads(line) for line in (target/'stdout.log').read_text().splitlines() if line.startswith('{')]
    if events!=[{'event':'resident','requests':count,'bulk_rows':rows}]:
        raise ValueError('resident role did not report the complete request count')
    if indices is None:
        indices=[index for index,prompt in enumerate(plan['prompts'])
                 if role!='control' or prompt['file'] in plan['kl_files']]
    check_tree(target,('stdout.log','stderr.log','run.json',*(str(index) for index in indices)))
    records={}
    for index in indices:
        prompt=plan['prompts'][index];root=target/str(index)
        clipped=prompt['file'] in plan['kl_files']
        check_tree(root,('native','clipped') if clipped else ('native',))
        records[index]={role:evidence(root/'native',record,prompt,rows,args.long_tokens,None)}
        if clipped:
            points=out/f'{index}-{role}-positions.ids'
            records[index][role+'-clipped']=evidence(root/'clipped',record,prompt,rows,args.long_tokens,points)
    if any(g.digest(Path(name))!=value for name,value in inputs.items()):
        raise ValueError('resident manifest or prompt inputs changed during capture')
    return records,{**record,'input_hashes':inputs}


def verify(plan,prepared):
    for role,probe in prepared['probes'].items():
        if identity(Path(probe['path']),Path(plan[role]))!=probe:
            raise ValueError('resident probe changed during capture')
    if any(g.digest(g.ROOT/'tools'/name)!=value for name,value in prepared['tool_hashes'].items()):
        raise ValueError('resident tools changed during capture')
    for group in plan.get('resident_groups',{}).values():
        if g.digest(Path(group['stdout_path']))!=group['stdout_sha256']:
            raise ValueError('resident group stdout changed after capture')
        if any(g.digest(Path(name))!=value for name,value in group['input_hashes'].items()):
            raise ValueError('resident manifest or input changed after capture')


def collect(args,plan,out,prepared):
    records={}
    for selected in prepared['groups']:
        role=selected['role']
        measured,group=collect_role(args,plan,out,prepared,role,selected['indices'],selected['name'])
        plan.setdefault('resident_groups',{})[selected['name']]=group
        for index,values in measured.items():
            records[index]={**records.get(index,{}),**values}
            for name,value in values.items():plan['captures'][f'{index}-{name}']=value
        verify(plan,prepared)
        g.save(out/'long-report.json',plan)
    return records
