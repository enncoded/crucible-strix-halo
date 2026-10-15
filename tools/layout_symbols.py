"""Symbol-use rule for engine/ and kernels/.

engine.hip is one translation unit, so a header can use a symbol of a higher
module without any #include. The include test (tools/layout.py) cannot see
that. This scan reads the top-level names that each file defines (types,
functions, variables, aliases, macros) and the identifiers that each file
uses. A file breaks the rule when it uses a name that is defined only in
modules of a higher rank than its own.

The scan is a lexical approximation, not a C++ parser. A name that the file
declares itself (a local, parameter or member) is dropped from the file's
uses, so a local that shares its name with a higher top-level name HIDES every
real use of that name in the file: a missed use, not a false hit. Keep local
names distinct from the names of higher modules. Do not widen the scan.
"""
from __future__ import annotations

import re
from pathlib import Path

import layout

IDENT = re.compile(r'[A-Za-z_]\w*')
DEFINE = re.compile(r'^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)', re.M)
TYPE_HEAD = re.compile(r'\b(?:struct|class|union|enum(?:[ \t]+(?:class|struct))?)\s+(?:\[\[[^\]]*\]\]\s*)?([A-Za-z_]\w*)')
KEYWORDS = frozenset({
    'if', 'for', 'while', 'switch', 'return', 'sizeof', 'alignof', 'static_assert', 'decltype',
    'operator', 'template', 'typename', 'noexcept', 'requires', 'main', 'catch', 'defined',
})


def strip_code(text: str) -> str:
    """TEXT without comments, string and character literals, and preprocessor lines.

    Each removed character becomes a space (newlines stay), so offsets hold.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        two = text[i:i + 2]
        if two == '//' or two == '/*':
            end = text.find('\n', i) if two == '//' else text.find('*/', i + 2)
            end = n if end < 0 else (end if two == '//' else end + 2)
            out.append(re.sub(r'[^\n]', ' ', text[i:end]))
            i = end
        elif c == '"' and i and text[i - 1] == 'R' and re.match(r'"([^(\s]{0,16})\(', text[i:]):
            close = ')' + re.match(r'"([^(\s]*)\(', text[i:]).group(1) + '"'
            end = text.find(close, i)
            end = n if end < 0 else end + len(close)
            out.append(re.sub(r'[^\n]', ' ', text[i:end]))
            i = end
        elif c in '"\'':
            j = i + 1
            while j < n and text[j] != c and text[j] != '\n':
                j += 2 if text[j] == '\\' else 1
            out.append(' ' * (min(j, n - 1) + 1 - i))
            i = min(j, n - 1) + 1
        else:
            out.append(c)
            i += 1
    code = ''.join(out)
    return re.sub(r'^[ \t]*#(?:[^\n\\]|\\\n?)*', lambda m: re.sub(r'[^\n]', ' ', m.group(0)),
                  code, flags=re.M)


def _header_names(header: str, body: bool = True) -> set[str]:
    """Names that one top-level statement HEADER (the text before its { or ;) defines.

    BODY is False when a ; ends the statement: `struct Name;` only declares.
    """
    header = re.sub(r'\btemplate\s*<[^;{]*?>\s*(?=[A-Za-z_])', ' ', header)
    flat = header.strip()
    if not flat or flat.startswith(('namespace', 'using namespace', 'extern', 'static_assert', 'friend')):
        return set()
    paren = flat.find('(')
    eq = flat.find('=')
    if re.match(r'(?:using|typedef)\b', flat):
        m = re.match(r'using\s+([A-Za-z_]\w*)\s*=', flat)
        if m:
            return {m.group(1)}
        names = IDENT.findall(re.sub(r'\([^)]*\)', ' ', flat))
        return {names[-1]} if names else set()
    types = TYPE_HEAD.search(flat)
    if types and body and (paren < 0 or types.start() < paren) and (eq < 0 or types.start() < eq):
        return {types.group(1)}
    if 0 <= eq and (paren < 0 or eq < paren):
        names = IDENT.findall(re.sub(r'\[[^\]]*\]', ' ', flat[:eq]))
        return {names[-1]} if names else set()
    if paren > 0:
        before = IDENT.findall(flat[:paren])
        if before and before[-1] not in KEYWORDS and 'operator' not in before:
            return {before[-1]}
    return set()


def defined_symbols(text: str) -> set[str]:
    """Names defined at namespace scope: types, functions, variables, aliases, macros."""
    code = strip_code(text)
    names = set(DEFINE.findall(text))
    stack = []  # 'ns' for a namespace or extern block, 'body' for any other brace
    start = 0
    i, n = 0, len(code)
    while i < n:
        c = code[i]
        if c == '{':
            header = code[start:i]
            top = all(s == 'ns' for s in stack)
            is_ns = bool(re.match(r'\s*(?:inline\s+)?(?:namespace\b[^=;(]*|extern\s*)$', header))
            if top and not is_ns:
                names |= _header_names(header)
            stack.append('ns' if is_ns else 'body')
            if top and not is_ns and '=' in header and '(' not in header.split('=')[0]:
                i = _skip_braces(code, i)  # an initializer: the statement runs on to ';'
                stack.pop()
                continue
            start = i + 1
        elif c == '}':
            if stack:
                stack.pop()
            start = i + 1
        elif c == ';':
            if all(s == 'ns' for s in stack):
                names |= _header_names(code[start:i], body=False)
            start = i + 1
        i += 1
    return {name for name in names if name not in KEYWORDS}


def _skip_braces(code: str, i: int) -> int:
    """Index after the brace block that opens at CODE[I]."""
    depth = 0
    while i < len(code):
        depth += (code[i] == '{') - (code[i] == '}')
        i += 1
        if depth == 0:
            break
    return i


# A name in declaration position: after a type word, or after a type that ends in * or &.
DECLARED = re.compile(
    r'(?:\b(?P<word>\w+)\s+|[A-Za-z_>\]][&*]+\s+|\w\s+[&*]+)(?P<name>[A-Za-z_]\w*)\s*(?=[({=;\[,):>])')
NOT_A_TYPE = frozenset({
    'return', 'else', 'new', 'throw', 'case', 'delete', 'goto', 'do', 'co_return', 'co_yield',
    'co_await', 'typename', 'sizeof', 'not', 'and', 'or', 'using', 'namespace'})


def used_identifiers(text: str) -> set[str]:
    """Identifiers that the file uses from outside.

    Member names (after . or ->) and names that the file declares itself
    (members, parameters, locals: `Type Name` followed by ( { = ; [ , ) or :)
    are left out, because they only share a spelling with a top-level name.
    """
    code = strip_code(text)
    declared = {m.group('name') for m in DECLARED.finditer(code)
                if m.group('word') not in NOT_A_TYPE}
    return {m.group(0) for m in IDENT.finditer(code)
            if m.group(0) not in declared
            and not re.search(r'(?:\.|->)\s*$', code[max(0, m.start() - 3):m.start()])}


def symbol_rank(path: str, lay: layout.Layout) -> int | None:
    """Rank of PATH's module; leaves count as -1 (any module may use them)."""
    row = lay.row(path)
    if row is None:
        return None
    return -1 if row.new in layout.LEAVES else layout.RANKS[row.module]


def violations(root: Path = layout.ROOT, lay: layout.Layout | None = None,
               tracked: frozenset[str] | None = None) -> list[str]:
    """Every use of a name defined only in a higher module, as 'symbol: src uses NAME (dst)'."""
    lay = lay or layout.Layout(layout.load_map(root))
    tracked = tracked if tracked is not None else layout.tracked_files(root)
    files = sorted(p for p in tracked if layout.is_source(p) and lay.row(p))
    texts = {p: (root / p).read_text(encoding='utf-8', errors='replace') for p in files}
    defs: dict[str, set[str]] = {}
    for path, text in texts.items():
        for name in defined_symbols(text):
            defs.setdefault(name, set()).add(lay.row(path).new)
    ranks = {lay.row(p).new: symbol_rank(p, lay) for p in files}
    found = set()
    for path, text in texts.items():
        src = lay.row(path)
        for name in used_identifiers(text) & defs.keys():
            homes = defs[name]
            if src.new in homes or min(ranks[h] for h in homes) <= ranks[src.new]:
                continue
            found.add(f'symbol: {src.new} uses {name} ({sorted(homes, key=lambda h: (ranks[h], h))[0]})')
    return sorted(found)
