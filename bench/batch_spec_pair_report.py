"""Validate mirrored same-load speculation trials. Diagnostic, not gate evidence."""
import argparse
import json
import math
import hashlib
import struct
from pathlib import Path
from statistics import median

ORDER = ('a', 'b', 'c', 'c', 'b', 'a')

def summarize(rows, cases):
    if len(rows) != len(ORDER) or tuple(row.get('case') for row in rows) != ORDER:
        raise ValueError('require mirrored a,b,c,c,b,a cases')
    tokens = rows[0].get('tokens')
    if type(tokens) is not int or tokens < 2 or tokens % 2 or any(row.get('tokens') != tokens for row in rows):
        raise ValueError('require equal positive token budgets')
    for row in rows:
        seconds = row.get('seconds')
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('invalid case duration')
        if type(row.get('steps')) is not int or row['steps'] <= 0:
            raise ValueError('invalid case step count')
    expected_dirs = {f'{index}_{case}' for index, case in enumerate(ORDER)}
    cases = Path(cases)
    if cases.is_symlink() or not cases.is_dir() or {path.name for path in cases.iterdir()} != expected_dirs:
        raise ValueError('require exactly six case directories')
    reference = {}
    for index, case in enumerate(ORDER):
        folder = cases / f'{index}_{case}'
        if folder.is_symlink() or not folder.is_dir(): raise ValueError('invalid case directory')
        for slot in range(2):
            path = folder / f'ids_{slot}.i32'
            if path.is_symlink() or not path.is_file(): raise ValueError('missing case IDs')
            data = path.read_bytes()
            if len(data) != 4 * (tokens // 2 + 1): raise ValueError('wrong ID count')
            if any(not 0 <= value[0] < 248320 for value in struct.iter_unpack('<i', data)):
                raise ValueError('invalid token ID')
            if index and data != reference[slot]: raise ValueError('case IDs differ')
            reference[slot] = data
    rates = {case: median(tokens / row['seconds'] for row in rows if row['case'] == case) for case in ('a', 'b', 'c')}
    paired = [dict(a_over_b=rows[b]['seconds']/rows[a]['seconds'], a_over_c=rows[c]['seconds']/rows[a]['seconds'])
              for a, b, c in ((0, 1, 2), (5, 4, 3))]
    return dict(status='PASS', rates=rates, a_over_b=rates['a']/rates['b'], a_over_c=rates['a']/rates['c'],
                paired_ratios=paired, clear_win_margin=0.05,
                clear_win=all(min(pair.values()) > 1.05 for pair in paired), observations=rows,
                ids_sha256={str(slot): hashlib.sha256(data).hexdigest() for slot, data in reference.items()})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a new report path.')
    rows = [json.loads(line[5:]) for line in args.log.read_text().splitlines() if line.startswith('PAIR ')]
    args.output.write_text(json.dumps(summarize(rows, args.cases), indent=2)+'\n')
