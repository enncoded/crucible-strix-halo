#!/usr/bin/env python3
"""Read the switch registry engine/switches.h for tests and gate tools; print the switch counts."""
from pathlib import Path
import re
import sys
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / 'engine/switches.h'
KINDS = {'kNumerics': 'numerics', 'kExactSpeed': 'exact-speed', 'kLayout': 'layout', 'kDiagnostic': 'diagnostic',
         'kBuild': 'build', 'kProfile': 'profile'}
USES = {'N': (False, False), 'S': (True, False), 'R': (False, True), 'SR': (True, True)}
ENTRY = re.compile(r'^\s*\{"(CRUCIBLE_[A-Z0-9_]+)", K::(k\w+), "([^"]*)", "(\w+)", (\d+), (\d+), (N|S|R|SR), '
                   r'"((?:[^"\\]|\\.)*)"\},\s*$')


class Switch(NamedTuple):
    name: str
    kind: str
    default: str
    module: str
    issue: int
    decide_by: int
    serve: bool
    record: bool
    note: str


def table_lines(text):
    body = text[text.index('inline constexpr SwitchInfo kSwitches[] = {'):]
    body = body[body.index('\n') + 1:body.index('\n};')]
    return [line for line in body.split('\n') if line.strip() and not line.strip().startswith('//')]


def load(header=HEADER):
    """Every kSwitches entry in table order; a line that does not parse is an error."""
    switches = []
    for line in table_lines(Path(header).read_text()):
        match = ENTRY.match(line)
        if not match:
            raise ValueError(f'cannot parse switch registry line: {line.strip()}')
        name, kind, default, module, issue, decide_by, use, note = match.groups()
        serve, record = USES[use]
        switches.append(Switch(name, KINDS[kind], default, module, int(issue), int(decide_by), serve, record, note))
    return switches


def external(header=HEADER):
    text = Path(header).read_text()
    body = text[text.index('kExternalVariables[] = {'):]
    return re.findall(r'"([A-Z0-9_]+)"', body[:body.index('};')])


def counts(switches):
    live = [s for s in switches if s.kind != 'build']
    return {'names': len(switches), 'live': len(live), 'build': len(switches) - len(live),
            **{kind: sum(s.kind == kind for s in switches) for kind in KINDS.values()}}


if __name__ == '__main__':
    result = counts(load(sys.argv[1] if len(sys.argv) > 1 else HEADER))
    print(' '.join(f'{key}={value}' for key, value in result.items()))
