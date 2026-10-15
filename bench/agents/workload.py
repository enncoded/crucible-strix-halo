"""Public, deterministic tool histories. Timed output never changes the next prompt."""
import hashlib
import json
from bench.suite.conversation_generate import TOOLS, SYSTEM, code

SUFFIX_SIZES = (256, 1024, 4096, 16384)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def common_prefix(left, right):
    return next((index for index, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


def nearest_text(target, stage, encode):
    low, high = 0, 1
    while len(encode(code(high, stage))) < target:
        high *= 2
        if high > 131072:
            raise ValueError('The public fixture exceeded its size bound.')
    while high - low > 1:
        middle = (low + high) // 2
        if len(encode(code(middle, stage))) < target:
            low = middle
        else:
            high = middle
    candidates = [code(count, stage) for count in (low, high)]
    return min(candidates, key=lambda text: abs(len(encode(text)) - target))


def body_for(messages, seed, output, metrics):
    return {'messages': messages, 'tools': TOOLS, 'tool_choice': 'none', 'enable_thinking': False,
            'max_tokens': output, 'stream': True, 'stream_options': {'include_usage': True},
            'return_token_ids': True, 'temperature': 0, 'seed': seed, 'spec': 0,
            'cache_control': 'auto', **({'crucible_metrics': True} if metrics else {})}


def extension(session_id, turn, text):
    call_id = f'synthetic-{session_id}-{turn}'
    return [{'role': 'assistant', 'content': 'I will inspect the public test output.',
             'tool_calls': [{'id': call_id, 'type': 'function', 'function': {
                 'name': 'read_file', 'arguments': json.dumps({'path': f'public/session-{session_id}/tests-{turn}.py'})}}]},
            {'role': 'tool', 'tool_call_id': call_id, 'name': 'read_file', 'content': text},
            {'role': 'user', 'content': 'Explain the next safe change. Name the tests. Use plain text, not tool calls.'}]


def admission(context, prefix, output, seed, suffixes, session_id):
    if (type(context) is not int or not 512 <= context <= 131072 or
            type(prefix) is not int or not 1 <= prefix < context - output or
            type(output) is not int or not 200 <= output <= 800 or
            type(seed) is not int or not 0 <= seed < (1 << 64) or
            type(session_id) is not int or not 0 <= session_id < 8 or
            not suffixes or any(type(value) is not int or not 1 <= value <= 16384 for value in suffixes)):
        raise ValueError('Invalid context, prefix, output, seed, suffix size, or session ID.')


def fitted_turn(history, session_id, stage, text, encode_request, encode_text, seed, output, metrics, context):
    def make(value):
        messages = [*history, *extension(session_id, stage, value)]
        body = body_for(messages, seed, output, metrics)
        return messages, body, encode_request(body)
    messages, body, ids = make(text)
    clipped = len(ids) + output > context
    if clipped:
        base = code(0, stage)
        _, _, minimum = make(base)
        remaining = context - output - len(minimum) + len(encode_text(base))
        if remaining <= len(encode_text(base)):
            return None
        for target in range(remaining, 0, -32):
            text = nearest_text(target, stage, encode_text)
            messages, body, ids = make(text)
            if len(ids) + output <= context:
                break
        else:
            return None
    return messages, body, ids, text, clipped


def session(encode_request, encode_text, *, session=0, seed=20261004, context_tokens=65536,
            prefix_tokens=12288, suffix_sizes=SUFFIX_SIZES, output_tokens=400, metrics=True):
    admission(context_tokens, prefix_tokens, output_tokens, seed, suffix_sizes, session)
    system = {'role': 'system', 'content': SYSTEM + '\nPublic repository context:\n' + nearest_text(prefix_tokens, 0, encode_text)}
    history, previous, result = [system], [], []
    answer = nearest_text(output_tokens, 1, encode_text)
    for stage in range(1024):
        target = suffix_sizes[stage % len(suffix_sizes)]
        text = nearest_text(target, seed + session * 1024 + stage, encode_text)
        fitted = fitted_turn(history, session, stage, text, encode_request, encode_text,
                             (seed + session) % (1 << 64), output_tokens, metrics, context_tokens)
        if fitted is None:
            break
        messages, body, ids, text, clipped = fitted
        if not ids or any(type(token) is not int or not 0 <= token < 248320 for token in ids):
            raise ValueError('The encoder returned invalid IDs.')
        result = [*result, {'session': session, 'turn': stage, 'body': body, 'request_sha256': digest(body),
                           'prompt_ids_sha256': digest(ids), 'prompt_tokens': len(ids),
                           'suffix_target_tokens': target, 'suffix_tokens': len(encode_text(text)), 'context_tail': clipped,
                           'context_limit_tokens': context_tokens, 'shared_previous_tokens': common_prefix(previous, ids),
                           'fixture_label': 'new_shared_prefix_session' if stage == 0 else 'continued_session',
                           'teacher_forced': True}]
        history = [*messages, {'role': 'assistant', 'content': answer}]
        previous = ids
    if not result:
        raise ValueError('No turn fits the requested context. Reduce the prefix or output budget.')
    return result
