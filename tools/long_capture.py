"""Capture long final logits and the approved clipped diagnostic samples."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import gates as g
import long_gates as lg
from build_long_probe import PROBE_FILES


def probe_identity(probe, engine, probe_files=PROBE_FILES, origin=lg.PROBE_ORIGIN):
    probe = probe.resolve(strict=True)
    path = Path(str(probe)+'.build.json')
    record = json.loads(path.read_text())
    expected = {name:g.digest(g.ROOT/name) for name in probe_files}
    if (record.get('capture_origin') != origin or record.get('probe_hashes') != expected or
            record.get('binary_sha256') != g.digest(probe) or
            record.get('bound_engine_binary_sha256') != g.digest(engine) or
            not g.layout.source_keys({'engine/engine.hip','engine/cli/gen_cli.h','engine/exl3/safetensors.h'}) <=
            g.layout.source_keys(record.get('source_hashes',{}))):
        raise ValueError('probe source, binary, or selected-engine binding differs from its build record')
    return {'path':str(probe),'binary_sha256':g.digest(probe),'build_record_sha256':g.digest(path),'build':record}


def usage(binary):
    done = subprocess.run([str(binary)],capture_output=True,text=True,timeout=30,check=False)
    text = done.stdout+done.stderr
    if done.returncode != 2 or any(not re.search(r'(?:^|\s|\[)'+flag+r'(?:\s|=|\]|$)',text)
                                    for flag in ('--dump-logits','--prefill')):
        raise ValueError('long engine usage must advertise --dump-logits and --prefill')
    return {'returncode':done.returncode,'command':[str(binary)],'usage_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'native_draft_head':'--draft-head full|64k' in text,
            'mtp_prefix_capable':bool(re.search(r'(?:^|\s|\[)--mtp-prefix(?:\s|=|\]|$)',text))}


def prompts(files):
    result, seen = (), set()
    for file in files:
        file = file.resolve(strict=True)
        payload = file.read_bytes()
        ids = tuple(g.parse_ids(payload.decode()))
        if len(ids) not in lg.LENGTHS or ids in seen:
            raise ValueError('long files need unique prompts of exactly 1K,4K,16K,64K,or128K IDs')
        seen = {*seen,ids}
        result = (*result,{'file':str(file),'tokens':len(ids),'sha256':hashlib.sha256(payload).hexdigest()})
    return result


def prepare(args, candidate):
    bitwise = prompts(getattr(args,'long_bitwise_ids_files',None) or [])
    kl = prompts(getattr(args,'long_kl_ids_files',None) or [])
    if not bitwise and not kl: raise ValueError('long options need explicit bitwise or KL prompt files')
    if bitwise and not {1024,4096,16384} <= {item['tokens'] for item in bitwise}:
        raise ValueError('long exact mode requires 1K, 4K, and 16K prompts')
    manifest = g.ROOT/'bench/suite/kl/manifest.json'
    fixed = json.loads(manifest.read_text())['prompts']
    if kl and (len(kl)!=7 or {item['sha256'] for item in kl} != set(fixed.values())):
        raise ValueError('KL needs all seven immutable prompts from bench/suite/kl/manifest.json')
    baseline = getattr(args,'long_baseline_engine',None) or args.baseline_engine
    name = getattr(args,'long_baseline_name',None) or (args.baseline_name if args.baseline_engine else '')
    if (baseline is None or not name.strip() or
            (getattr(args,'long_baseline_engine',None) is not None and not getattr(args,'long_baseline_name',None))):
        raise ValueError('long checks need a named independent baseline engine')
    baseline = baseline.resolve(strict=True)
    if g.digest(baseline) == g.digest(candidate): raise ValueError('long candidate cannot serve as its own baseline')
    rows = getattr(args,'long_prefill_rows',4096)
    maximum = getattr(args,'kl_max',None)
    tokens = getattr(args,'long_tokens',args.tokens)
    if type(rows) is not int or rows < 1: raise ValueError('long prefill rows must be a positive integer')
    if type(tokens) is not int or not 1 <= tokens <= 4096: raise ValueError('long tokens must be an integer in [1,4096]')
    lg.assess([{'prompt':'validation','candidate':{'forward':0.,'reverse':0.,'top1':0},
                'baseline':{'forward':0.,'reverse':0.,'top1':0,'margin':1.},'production':False,'control_top1':0}],maximum)
    combined = {item['file']:item for item in (*bitwise,*kl)}
    probe_paths = (getattr(args,'long_candidate_probe',None),getattr(args,'long_baseline_probe',None))
    resident_requested=any(getattr(args,name,None) is not None for name in
                           ('long_resident_candidate_probe','long_resident_baseline_probe'))
    if not resident_requested and (getattr(args,'long_resident_shard','none')!='none' or
                                   getattr(args,'long_resident_16k_control_timeout',300)!=300):
        raise ValueError('resident sharding needs both source-bound probes')
    if resident_requested and any(path is not None for path in probe_paths):
        raise ValueError('choose standalone or resident probes, not both')
    if kl and not resident_requested and any(path is None for path in probe_paths):
        raise ValueError('sampled KL needs candidate and baseline diagnostic probes')
    plan = {'status':'FAIL','baseline':str(baseline),'baseline_name':name,'candidate':str(candidate),
            'candidate_sha256':g.digest(candidate),'baseline_sha256':g.digest(baseline),
            'usage':{'candidate':usage(candidate),'baseline':usage(baseline)},'prefill_rows':rows,
            'tokens':getattr(args,'long_tokens',args.tokens),'absolute_kl_max':maximum,
            'bitwise_files':[item['file'] for item in bitwise],'kl_files':[item['file'] for item in kl],
            'prompts':list(combined.values()),'captures':{},'capture_origin':lg.PROBE_ORIGIN,
            'candidate_probe':probe_identity(probe_paths[0],candidate) if kl and not resident_requested else None,
            'baseline_probe':probe_identity(probe_paths[1],baseline) if kl and not resident_requested else None,
            'fixed_corpus_sha256':g.digest(manifest),
            'tool_hashes':{name:g.digest(g.ROOT/'tools'/name) for name in ('gates.py','long_capture.py','long_gates.py','long_run.py','build_long_probe.py')}}
    if resident_requested:
        import resident_capture
        plan['resident']=resident_capture.prepare(args,plan)
        plan['capture_origin']=plan['resident']['origin']
    return plan


def control_exception():
    return {'scope':'16K small-row control only','approved_by':'maintainer',
            'prompt_tokens':16384,'prefill_rows':0,'priority':2,'timeout_seconds':600}


def validate_control_exception(command,args,exception):
    if exception!=control_exception() or args.priority!=2 or len(command)!=5 or command[3]!='0':
        raise ValueError('invalid 16K control timeout exception')
    lines=Path(command[-1]).read_text().splitlines()
    fields=lines[0].split('\t') if len(lines)==1 else []
    if len(fields)!=4 or fields[1]=='-' or prompts([Path(fields[0])])[0]['tokens']!=16384:
        raise ValueError('the timeout exception requires one 16K native-plus-clipped control request')


def floor_command(command, out):
    """The memory floor inside the lease: mem_floor.sh polls
    MemAvailable as lease_runner.wait_memory does, then execs the engine; below the floor it exits 76 with no engine."""
    return ['env',f'MEM_FLOOR_WAIT_S={g.LEASE_MEM_WAIT_S}',f'MEM_FLOOR_POLL_S={g.LEASE_MEM_POLL_S}','bash',
            str(g.ROOT/'tools/mem_floor.sh'),str(g.LEASE_MEM_FLOOR_GB),str(out/'memory.log'),'--',*command]


def below_floor(returncode, out):
    """Exit 76 with a 'below: no start' last line: no engine ran, so the lease runs again."""
    log = out/'memory.log'
    lines = log.read_text().splitlines() if log.exists() else []
    return returncode == g.LEASE_EXIT_MEMORY and bool(lines) and 'below: no start' in lines[-1]


def execute(command, out, args, *, timeout_exception=None, reference=False):
    if timeout_exception is not None:validate_control_exception(command,args,timeout_exception)
    budget=600 if timeout_exception else 300
    out.mkdir()
    queue_wait = getattr(args,'queue_wait',g.QUEUE_WAIT_S)
    queued = [str(args.gpu_run),'-p',str(args.priority),'-w',str(queue_wait),'--','timeout','-k','10',
              str(budget+g.LEASE_MEM_WAIT_S),*floor_command(command,out)]
    env = g.capture_environment(reference)
    record = {'command':command,'queued_command':queued,'started_ns':time.time_ns(),
              'environment':g.runtime_environment(env),'timeout_seconds':budget,
              'memory_wait_seconds':g.LEASE_MEM_WAIT_S,
              **({'timeout_exception':dict(timeout_exception)} if timeout_exception else {})}
    g.save(out/'run.json',record)
    latest_ns, retries = time.monotonic_ns()+queue_wait*10**9, 0
    while True:
        with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
            done = subprocess.run(queued,stdout=stdout,stderr=stderr,env=env,check=False)
        if not below_floor(done.returncode,out):
            break
        retries += 1
        if time.monotonic_ns() >= latest_ns:
            raise ValueError(f'long capture: the memory floor of {g.LEASE_MEM_FLOOR_GB} GB did not come before the '
                             f'queue wait of {queue_wait} s ended; inspect {out}/memory.log')
    record = {**record,'returncode':done.returncode,'ended_ns':time.time_ns(),'memory_retries':retries}
    g.save(out/'run.json',record)
    if done.returncode: raise ValueError(f'long capture failed ({done.returncode}); inspect {out}')
    g.regular(out/'stdout.log',record['started_ns'])
    return {**record,'stdout_sha256':g.digest(out/'stdout.log'),'stdout_path':str(out/'stdout.log')}


def native(binary, prompt, out, rows, args, candidate_policy):
    tokens = getattr(args,'long_tokens',args.tokens)
    command = [str(binary),'gen',str(args.checkpoint.resolve()),'--ids-file',prompt['file'],
               '--n',str(tokens),'--ctx',str(max(32768,prompt['tokens']+tokens)),
               '--chunk','8','--spec','0','--prefill',str(rows),'--dump-logits',str(out/'logits.f32'),
               '--draft-q','0','--draft-adapt','0','--draft-margin','0','--lookup','0',
               '--draft-overlap','0','--draft-last-head','1']
    if candidate_policy is not None and candidate_policy['native']:
        command += ['--draft-head','full','--draft-head-auto','0']
    if candidate_policy is not None and candidate_policy.get('mtp_prefix_capable',False):
        cap = getattr(args,'mtp_prefix',None) if binary.resolve()==args.engine.resolve() else None
        command += ['--mtp-prefix',str(0 if cap is None else cap)]
    record = execute(command,out,args,reference=binary.resolve()!=args.engine.resolve())
    return {**record,**native_evidence(out/'logits.f32',out/'stdout.log',record['started_ns'],tokens)}


def native_evidence(logit_path,stdout_path,started_ns,tokens):
    logits = lg.check_logits(logit_path,1,started_ns)
    values = next(lg.rows(logit_path,1))
    generated = g.parse_output(stdout_path.read_text(),tokens)
    if max(range(lg.VOCAB),key=values.__getitem__) != generated[0]:
        raise ValueError('native copied logits disagree with the first target ID')
    return {'kind':'production schedule','logits':logits,'logit_path':str(logit_path),
            'output_ids':generated,'point_count':1}


def clipped(probe, prompt, out, rows, args, *, reference=False):
    points = lg.sample_points(prompt['tokens'])
    points_file = out.parent/(out.name+'-positions.ids')
    points_file.write_text(','.join(map(str,points))+'\n')
    command = [probe['path'],str(args.checkpoint.resolve()),prompt['file'],str(points_file),str(rows),
               str(max(32768,prompt['tokens']+args.tokens)),str(out/'data')]
    record = execute(command,out,args,reference=reference)
    if (out/'data').is_symlink() or not (out/'data').is_dir() or {file.name for file in (out/'data').iterdir()} != {'logits.f32'}:
        raise ValueError('clipped capture must contain only the canonical logits file')
    evidence=clipped_evidence(out/'data/logits.f32',out/'stdout.log',record['started_ns'],prompt,rows)
    return {**record,**evidence,'positions_path':str(points_file),'positions_sha256':g.digest(points_file)}


def clipped_evidence(logit_path,stdout_path,started_ns,prompt,rows):
    points=lg.sample_points(prompt['tokens'])
    plan=lg.schedule(prompt['tokens'],points,rows if rows else 8)
    events = [json.loads(line) for line in stdout_path.read_text().splitlines() if line.startswith('{')]
    if any(not isinstance(event,dict) for event in events): raise ValueError('invalid clipped event object')
    samples = [event for event in events if event.get('event')=='point']
    schedules = [event for event in events if event.get('event')=='schedule']
    if (len(events) != len(points)+1 or len(schedules) != 1 or schedules[0].get('kind') != 'clipped schedule' or
            schedules[0].get('blocks') != [list(block) for block in plan] or
            any(type(value) is not int for block in schedules[0].get('blocks',[]) for value in block) or
            [event.get('position') for event in samples] != list(points) or
            any(type(event.get('position')) is not int for event in samples) or
            any(event.get('copied_argmax') is not True or type(event.get('argmax')) is not int for event in samples)):
        raise ValueError('clipped producer schedule or complete argmax proof differs')
    logits = lg.check_logits(logit_path,len(points),started_ns)
    for event,values in zip(samples,lg.rows(logit_path,len(points))):
        if max(range(lg.VOCAB),key=values.__getitem__) != event['argmax']:
            raise ValueError('clipped copied logits disagree with reported argmax')
    return {'kind':'clipped schedule','schedule':[list(block) for block in plan],
            'points':list(points),'argmax_ids':[event['argmax'] for event in samples],
            'logits':logits,'logit_path':str(logit_path),'point_count':len(points)}
