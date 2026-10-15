"""Long-prompt evidence. Sampled rows use an explicitly clipped schedule."""
import array
import hashlib
import math
from pathlib import Path
import sys
from statistics import median

import gates as g

VOCAB = g.VOCAB
RELATIVE_FACTOR = 3.0
MEDIAN_FLOOR = 1e-5
TIE_MARGIN = 1e-3
SAMPLE_STRIDE = 256
TAIL_ROWS = 64
LENGTHS = (1024,4096,16384,65536,131072)
PROBE_ORIGIN = 'q405b 4bf0d90:bench/quality_probe.hip'


def sample_points(length):
    if type(length) is not int or length < 1:
        raise ValueError('long prompt length must be positive')
    return tuple(sorted(set(range(SAMPLE_STRIDE-1,length,SAMPLE_STRIDE)) |
                        set(range(max(0,length-TAIL_ROWS),length))))


def schedule(length, points, max_rows):
    if (not points or tuple(sorted(set(points))) != tuple(points) or points[-1] != length-1 or
            any(type(point) is not int or not 0 <= point < length for point in points) or max_rows < 1):
        raise ValueError('clipped schedule needs sorted unique valid points ending at the last position')
    result, offset = (), 0
    for point in points:
        while offset <= point:
            count = min(max_rows,point+1-offset)
            result = (*result,(offset,count))
            offset += count
    return result


def check_logits(path, count, started_ns):
    info = g.regular(path,started_ns)
    if info.st_size != count*VOCAB*4:
        raise ValueError(f'logit capture needs exactly {count} full float32 vocabulary rows: {path}')
    return {'bytes':info.st_size,'sha256':g.digest(path),'mtime_ns':info.st_mtime_ns,'ctime_ns':info.st_ctime_ns}


def rows(path, count):
    with Path(path).open('rb') as stream:
        for _ in range(count):
            values = array.array('f')
            values.frombytes(stream.read(VOCAB*4))
            if sys.byteorder != 'little': values.byteswap()
            if len(values) != VOCAB or any(not math.isfinite(v) for v in values):
                raise ValueError('missing or nonfinite full-vocabulary logit row')
            yield values
        if stream.read(1):
            raise ValueError('extra logit data after the required rows')


def normalized(values):
    maximum = max(values)
    shifted = tuple(float(value)-maximum for value in values)
    denominator = math.log(math.fsum(math.exp(value) for value in shifted))
    return tuple(value-denominator for value in shifted)


def divergence(candidate, control):
    if (len(candidate) < 2 or len(candidate) != len(control) or
            any(not math.isfinite(v) for values in (candidate,control) for v in values)):
        raise ValueError('KL needs equally sized finite vocabulary vectors')
    a,b = normalized(candidate),normalized(control)
    forward = math.fsum(math.exp(x)*(x-y) for x,y in zip(a,b))
    reverse = math.fsum(math.exp(y)*(y-x) for x,y in zip(a,b))
    top = max(range(len(candidate)),key=candidate.__getitem__)
    second = max(value for index,value in enumerate(candidate) if index != top)
    if not math.isfinite(forward) or not math.isfinite(reverse) or min(forward,reverse) < -1e-10:
        raise ValueError('invalid KL result')
    return {'forward':max(0.,forward),'reverse':max(0.,reverse),'top1':top,'margin':candidate[top]-second}


def assess(points, absolute_max):
    if not points or (absolute_max is not None and (not math.isfinite(absolute_max) or absolute_max < 0)):
        raise ValueError('KL needs measured points and a finite nonnegative optional absolute limit')
    for point in points:
        for role in ('candidate','baseline'):
            if any(not math.isfinite(point[role][key]) or point[role][key] < 0 for key in ('forward','reverse')):
                raise ValueError('nonfinite or negative pointwise KL')
        if not math.isfinite(point['baseline']['margin']) or point['baseline']['margin'] < 0:
            raise ValueError('invalid baseline top-one margin')
    prompts = sorted({point['prompt'] for point in points})
    per_prompt = {prompt:{role:{direction:median(point[role][direction] for point in points if point['prompt']==prompt)
                               for direction in ('forward','reverse')} for role in ('candidate','baseline')}
                  for prompt in prompts}
    medians = {role:{direction:median(value[role][direction] for value in per_prompt.values())
                    for direction in ('forward','reverse')} for role in ('candidate','baseline')}
    limits = {key:RELATIVE_FACTOR*max(value,MEDIAN_FLOOR) for key,value in medians['baseline'].items()}
    mismatches = [point for point in points if point['candidate']['top1'] != point['baseline']['top1']]
    near_ties = [point for point in mismatches if point['baseline']['margin'] < TIE_MARGIN]
    bad_top1 = [point for point in mismatches if point['baseline']['margin'] >= TIE_MARGIN]
    passed = not bad_top1 and all(medians['candidate'][key] <= limit and
                                 (absolute_max is None or medians['candidate'][key] <= absolute_max)
                                 for key,limit in limits.items())
    return {'status':'PASS' if passed else 'FAIL','medians':medians,'per_prompt_medians':per_prompt,
            'limits':limits,'point_count':len(points),'points':points,
            'near_ties':near_ties,'top1_failures':bad_top1,
            'worst_five':sorted(points,key=lambda point:max(point['candidate']['forward'],point['candidate']['reverse']),reverse=True)[:5],
            'policy':{'relative_factor':RELATIVE_FACTOR,'median_floor':MEDIAN_FLOOR,'near_tie_margin':TIE_MARGIN,
                      'absolute_median_max':absolute_max,'per_point_bounds':False,
                      'aggregate':'median within each prompt, then median across prompts, separately for each direction',
                      'top1_reference':'current main, including every clipped sample',
                      'reason':'Prompt weighting must not depend on context length. Oracle Gate 1(b) remains binding.'}}
