#!/usr/bin/env python3
"""Build the approved diagnostic adapter against a recorded engine source."""
import argparse
import hashlib
import json
import shlex
import re
from pathlib import Path
import subprocess

import gates as g
from long_gates import PROBE_ORIGIN

PROBE_FILES = ('bench/long_probe.hip','bench/long_probe_runtime.h')


def source_hashes(source):
    files = sorted([*source.glob('engine/**/*'),*source.glob('kernels/**/*'),source/'engine/exl3/safetensors.h'])
    return {str(file.relative_to(source)):g.digest(file) for file in files if file.is_file()}


def compiler_flags(command, source=None, build_cwd=None):
    tokens = shlex.split(command) if isinstance(command,str) else command
    if not isinstance(tokens,list) or any(type(token) is not str for token in tokens):
        raise ValueError('engine build record needs its compiler command')
    # Silent option removal can make the probe's numerics differ from its engine.
    fixed = {'-std=c++20','--offload-arch=gfx1151','--rocm-path=/opt/rocm',
             '-I/opt/rocm/include','-lpthread','-g','-g0','-gline-tables-only'}
    numeric = {'fast-math','unsafe-math-optimizations','finite-math-only','associative-math',
               'reciprocal-math','signed-zeros','trapping-math','rounding-math','approx-func',
               'approx-transcendentals'}
    inherited = {f'-f{prefix}{flag}' for prefix in ('','no-') for flag in numeric}
    inherited |= {'-ffp-contract=off','-ffp-contract=on','-ffp-contract=fast'}
    selected = source.resolve() if source is not None else None
    cwd = Path(build_cwd).resolve() if build_cwd is not None else Path.cwd()
    result, index, source_count = (), 0, 0
    while index < len(tokens):
        token = tokens[index]
        if token in ('-D','-U','-o'):
            index += 1
            if index == len(tokens) or tokens[index].startswith('-'):
                raise ValueError('incomplete compiler option')
            if token=='-o':
                index += 1
                continue
            token += tokens[index]
        if token.startswith(('-D','-U')):
            if len(token)==2 or token.startswith(('-DCRUCIBLE_ENGINE_HEADER','-UCRUCIBLE_ENGINE_HEADER')):
                raise ValueError('invalid or reserved compiler definition')
            result = (*result,token)
        elif token in inherited or re.fullmatch(r'-O(?:[0-3sgz]|fast)',token):
            result = (*result,token)
        elif token in fixed or (index==0 and Path(token).name=='hipcc'):
            pass
        elif selected is not None and not token.startswith('-') and (cwd/token).resolve()==selected:
            source_count += 1
        else:
            raise ValueError(f'unaudited compiler option or input: {token}')
        index += 1
    if selected is not None and source_count!=1:
        raise ValueError('engine build command needs exactly one canonical selected engine source')
    if not any(re.fullmatch(r'-O(?:[0-3sgz]|fast)',flag) for flag in result):
        raise ValueError('selected compiler command needs an explicit optimization option')
    return result


def build(args,probe_files=PROBE_FILES,capture_origin=PROBE_ORIGIN):
    source,engine,output = args.source.resolve(),args.engine.resolve(),args.output.resolve()
    payload = args.engine_build_record.read_bytes()
    record_hash = hashlib.sha256(payload).hexdigest()
    record = json.loads(payload)
    flags = compiler_flags(record.get('command'),source/'engine/engine.hip',record.get('build_cwd'))
    snapshot = source_hashes(source)
    if not snapshot or record.get('source_hashes') != snapshot or record.get('binary_sha256') != g.digest(engine):
        raise ValueError('complete source and binary hashes must match the selected engine build record')
    if output.exists() or Path(str(output)+'.build.json').exists():
        raise ValueError('use a new probe binary path')
    hashes = {name:g.digest(g.ROOT/name) for name in probe_files}
    command = [args.hipcc,'-O3','-std=c++20','--offload-arch=gfx1151','--rocm-path=/opt/rocm',
               '-I/opt/rocm/include',*flags,f'-DCRUCIBLE_ENGINE_HEADER="{source}/engine/engine.hip"',
               str(g.ROOT/probe_files[0]),'-lpthread','-o',str(output)]
    output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(command,check=True)
    if not output.is_file() or output.stat().st_size == 0:
        raise ValueError('compiler did not create a probe binary')
    if (g.digest(args.engine_build_record) != record_hash or source_hashes(source) != snapshot or
            record['binary_sha256'] != g.digest(engine) or
            any(g.digest(g.ROOT/name) != value for name,value in hashes.items())):
        raise ValueError('source or selected engine changed during probe compilation')
    g.save(Path(str(output)+'.build.json'),{'schema':1,'capture_origin':capture_origin,
           'binary_sha256':g.digest(output),'bound_engine_binary_sha256':g.digest(engine),
           'engine_build_record_sha256':record_hash,'source_hashes':snapshot,'inherited_compiler_flags':flags,
           'bound_engine_source_path':str(source/'engine/engine.hip'),
           'source_commit':record.get('source_commit'),'source_branch':record.get('source_branch'),
           'probe_hashes':hashes,'command':command})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--engine-build-record',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hipcc',default='/opt/rocm/bin/hipcc')
    try: build(parser.parse_args())
    except (ValueError,OSError,subprocess.SubprocessError) as error: parser.exit(1,f'probe build: {error}\n')


if __name__ == '__main__': main()
