#!/usr/bin/env python3
"""Create an instrumented build copy. Do not change production source."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


def replace(source, old, new, count=1):
    actual = source.count(old)
    if actual != count:
        raise ValueError(f"expected {count} copies of {old!r}, found {actual}")
    return source.replace(old, new)


def engine_trace(source):
    source = '#include "../bench/explore/startup_trace.h"\n' + source
    source = replace(source, '    Loader(dir).Load(m_, streaming);',
                     '    const auto probe_load = ProbeNow();\n    Loader(dir).Load(m_, streaming);\n'
                     '    ProbePhase("model_load", probe_load);\n    const auto probe_buffers = ProbeNow();')
    source = replace(source, '    if (prefill_rows_) InitPrefill();',
                     '    if (prefill_rows_) InitPrefill();\n    HIP_OK(hipDeviceSynchronize());\n'
                     '    ProbePhase("buffer_init", probe_buffers);')
    old = '''    Forward(&token, 1, 0, 1);
    // The warmup plan pre-build is disabled until a correct fix exists: a
    // 4096-row plan built here faults on the first 4096-token prompt (GPU
    // page fault; train 3 gate bulk4k305). The first request builds it.
    Reset();'''
    new = '''    auto probe = ProbeNow();
    Forward(&token, 1, 0, 1);
    HIP_OK(hipDeviceSynchronize());
    ProbePhase("forward", probe);
    probe = ProbeNow();
    if (ProbeEager() && prefill_rows_) PrefillPlansFor(prefill_rows_);
    HIP_OK(hipDeviceSynchronize());
    ProbePhase("plan_build", probe);
    probe = ProbeNow();
    Reset();
    HIP_OK(hipDeviceSynchronize());
    ProbePhase("reset", probe);'''
    source = replace(source, old, new)
    return replace(source, '    WarmDeviceBuffers(spans);',
                   '    probe = ProbeNow();\n    WarmDeviceBuffers(spans);\n    ProbePhase("memory_touch", probe);')


def model_trace(source):
    calls = {
        'HIP_OK(hipHostMalloc(&stage, capacity));': ("PinAlloc", 2),
        'HIP_OK(hipHostFree(stage));': ("PinFree", 2),
        'HIP_OK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));': ("StreamCreate", 1),
        'HIP_OK(hipMemcpyAsync(t.dst + o, stage, n, hipMemcpyHostToDevice, st));': ("Copy", 1),
        'HIP_OK(hipMemcpyAsync(t.dst, stage, t.bytes, hipMemcpyHostToDevice, st));': ("Copy", 1),
        'HIP_OK(hipStreamSynchronize(st));': ("Sync", 2),
        'std::memcpy(stage, s + o, n);': ("Fill", 1),
        't.fill(stage);': ("Fill", 1),
    }
    for call, (bucket, count) in calls.items():
        source = replace(source, call, '{ ProbeTimer timer(ProbeBucket::' + bucket + '); ' + call + ' }', count)
    return replace(source, '      stage_live_.fetch_sub(capacity);\n    };',
                   '      stage_live_.fetch_sub(capacity);\n      ProbeWorkerEnd();\n    };')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    for directory in ('engine', 'kernels', 'server', 'bench'):
        shutil.copytree(root / directory, args.output / directory)
    patches, hashes = [], {}
    for relative, instrument in (('engine/engine.hip', engine_trace), ('engine/model/flashnext/model.h', model_trace)):
        source = (root / relative).read_text()
        traced = instrument(source)
        (args.output / relative).write_text(traced)
        hashes[relative] = {'original': hashlib.sha256(source.encode()).hexdigest(),
                            'instrumented': hashlib.sha256(traced.encode()).hexdigest()}
        patches.extend(difflib.unified_diff(source.splitlines(True), traced.splitlines(True),
                                           fromfile=relative, tofile=relative + '.probe'))
    (args.output / 'instrumentation.patch').write_text(''.join(patches))
    (args.output / 'source.json').write_text(json.dumps(
        {'source_sha': args.source_sha, 'files': hashes}, indent=2) + '\n')


if __name__ == '__main__':
    main()
