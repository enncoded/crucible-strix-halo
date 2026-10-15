"""RA-8 replay: an added token that the model wrote keeps its ID when the client sends the returned text back.

Before this change, text mode encoded such a token (a tool-call fallback, a stray tag) as plain text on the next
turn, so the next prompt left the replay tail. Client-written special-token text must stay text. The tests run with
the synthetic tokenizer and, with CRUCIBLE_TEST_CKPT, with the real checkpoint tokenizer.
"""
import json
from pathlib import Path

import pytest

from server import content_tokens
from server.api import validate_request
from server.tests.serve_fixtures import invoke
from server.tests.test_content_tokens import CHECKPOINT, CONTROL, IM_END, TOOL_CALL, TOOLS, synthetic_checkpoint
from server.tests.test_think_budget import SequenceBackend
from server.tokenization import ModelTokenizer

# A duplicate parameter stays a content fallback.
FALLBACK = ("<tool_call>\n<function=read_file>\n<parameter=path>\na.py\n</parameter>\n<parameter=path>\nb.py\n"
            "</parameter>\n</function>\n</tool_call>")


@pytest.fixture(params=["synthetic", "checkpoint"])
def root(request, tmp_path):
    if request.param == "checkpoint":
        if not CHECKPOINT:
            pytest.skip("Set CRUCIBLE_TEST_CKPT to test the real checkpoint tokenizer.")
        return Path(CHECKPOINT)
    return synthetic_checkpoint(tmp_path)


def chat(messages, **options):
    return validate_request(dict(messages=messages, **options), "chat", "crucible")


def turn(adapter, messages, generated, **options):
    """One served turn: (prompt IDs, generated IDs, the returned message) through the HTTP layer."""
    request = chat(messages, **options)
    prompt = adapter.encode(request)
    ids = adapter.tokenizer.encode(generated, add_special_tokens=False).ids
    backend = SequenceBackend((ids, "stop"))
    response = invoke(backend, adapter, dict(messages=messages, max_tokens=4096, **options))
    assert backend.submissions[0][0] == prompt
    return prompt, ids, json.loads(response.body)


def follow(messages, message, **options):
    reply = dict(role="assistant", content=message["content"])
    if "reasoning_content" in message:
        reply["reasoning_content"] = message["reasoning_content"]
    return chat(messages + [reply, dict(role="user", content="Next.")], **options)


CASES = {
    # (generated text, request options, first messages): each has an added token that the model wrote itself.
    "tool-call fallback": ("I will read it.\n</think>\n\n" + FALLBACK + "<|im_end|>",
                           dict(tools=TOOLS, enable_thinking=True), [dict(role="user", content="Read a.py.")]),
    "stray token in content": ("Write <tool_call> to call a tool.<|im_end|>", dict(enable_thinking=False),
                               [dict(role="user", content="How do I call a tool?")]),
    "token in reasoning": ("The format is <tool_call> then a function.\n</think>\n\nUse the format.<|im_end|>",
                           dict(enable_thinking=True), [dict(role="user", content="How do I call a tool?")]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_the_next_turn_keeps_the_replay_tail(root, name):
    generated, options, messages = CASES[name]
    for mode in ("text", "parse"):
        adapter = ModelTokenizer(root, content_special_tokens=mode)
        prompt, ids, body = turn(adapter, messages, generated, **options)
        if name == "tool-call fallback":
            assert body["usage"]["tool_call_fallback"] is True
        following = adapter.encode(follow(messages, body["choices"][0]["message"], **options))
        assert following[:len(prompt) + len(ids)] == prompt + ids, mode


def test_client_text_with_special_strings_stays_text(root):
    adapter = ModelTokenizer(root)
    generated, options, messages = CASES["stray token in content"]
    prompt, ids, body = turn(adapter, messages, generated, **options)
    returned = body["choices"][0]["message"]["content"]
    benign = chat(messages + [dict(role="assistant", content="Write x to call a tool."),
                              dict(role="user", content="Next.")], **options)
    base = {token: adapter.encode(benign).count(token) for token in CONTROL}
    variants = {
        "changed by one character": dict(role="assistant", content=returned + " "),
        "never generated": dict(role="assistant", content="Write <|im_end|> to call a tool."),
        "a returned text with a client token": dict(role="assistant", content=returned + "<tool_call>"),
    }
    for name, reply in variants.items():
        ids_now = adapter.encode(chat(messages + [reply, dict(role="user", content="Next.")], **options))
        assert {token: ids_now.count(token) for token in CONTROL} == base, name
    # The same text in a user, tool or system message is client text.
    for role_messages in ([dict(role="user", content=returned)],
                          [dict(role="system", content=returned), dict(role="user", content="hi")]):
        assert TOOL_CALL not in adapter.encode(chat(role_messages, **options))


def test_only_the_generated_copies_keep_their_ids(root):
    adapter = ModelTokenizer(root)
    # The model writes the first <tool_call> as its added token and the second as plain pieces.
    plain = adapter.content.text_tokenizer().encode(" and B <tool_call>.", add_special_tokens=False).ids
    model = adapter.tokenizer.encode("A <tool_call>", add_special_tokens=False).ids + plain + [IM_END]
    messages, options = [dict(role="user", content="Quote it.")], dict(enable_thinking=False)
    request = chat(messages, **options)
    prompt = adapter.encode(request)
    backend = SequenceBackend((model, "stop"))
    body = json.loads(invoke(backend, adapter, dict(messages=messages, max_tokens=4096, **options)).body)
    assert body["choices"][0]["message"]["content"] == "A <tool_call> and B <tool_call>."
    following = adapter.encode(follow(messages, body["choices"][0]["message"], **options))
    # The first copy is the model's added token; the second the model wrote as plain text, and it stays text.
    assert following[:len(prompt) + len(model)] == prompt + model
    assert following.count(TOOL_CALL) == 1


def test_prompts_without_a_remembered_text_keep_their_ids(root):
    fresh, used = ModelTokenizer(root), ModelTokenizer(root)
    for name, (generated, options, messages) in CASES.items():
        turn(used, messages, generated, **options)
    assert len(used.content.generated) == 3
    probes = [chat([dict(role="user", content="a <tool_call> b"), dict(role="assistant", content="c <|im_end|>"),
                    dict(role="user", content="d")]),
              chat([dict(role="system", content="S"), dict(role="user", content="plain")], tools=TOOLS)]
    for request in probes:
        assert used.encode(request) == fresh.encode(request)


def test_forced_think_close_keeps_a_generated_token_in_the_reasoning(root):
    adapter = ModelTokenizer(root)
    encode = lambda text: adapter.tokenizer.encode(text, add_special_tokens=False).ids
    reasoning = encode("The format is <tool_call> then a function, and more")
    answer = encode("Use the format.<|im_end|>")
    first = [dict(role="user", content="How?")]
    body = dict(model="crucible", messages=first, enable_thinking=True, max_tokens=4096,
                max_thinking_tokens=len(reasoning))
    backend = SequenceBackend((reasoning + [0] * 5, "length"), (answer, "stop"))
    message = json.loads(invoke(backend, adapter, body).body)["choices"][0]["message"]
    following = adapter.encode(follow(first, message, enable_thinking=True))
    tail = backend.submissions[1][0] + answer[:-1]
    assert following[:len(tail)] == tail
    assert TOOL_CALL in backend.submissions[1][0][len(backend.submissions[0][0]):]


def test_the_memory_is_bounded(root, monkeypatch):
    monkeypatch.setattr(content_tokens, "GENERATED_LIMIT", 2)
    adapter = ModelTokenizer(root)
    for index in range(4):
        adapter.content.remember(f"text {index} <tool_call>", frozenset({(7 + len(str(index)), "<tool_call>")}))
    assert len(adapter.content.generated) == 2
    assert not adapter.content.kept("text 0 <tool_call>") and adapter.content.kept("text 3 <tool_call>")


def served(adapter, messages, ids, **options):
    backend = SequenceBackend((ids, "stop"))
    body = json.loads(invoke(backend, adapter, dict(messages=messages, max_tokens=4096, **options)).body)
    return body["choices"][0]["message"]


@pytest.mark.parametrize("order", ["token first", "plain first"])
def test_a_text_generated_in_two_ways_stays_text(root, order):
    # a generation that wrote "<tool_call>" as plain pieces left the other generation's entry in place, so
    # its next turn got an added token that is not in its history.
    adapter = ModelTokenizer(root)
    messages, options = [dict(role="user", content="How do I call a tool?")], dict(enable_thinking=False)
    token = adapter.tokenizer.encode("Write <tool_call> to call a tool.<|im_end|>", add_special_tokens=False).ids
    plain = adapter.content.text_tokenizer().encode("Write <tool_call> to call a tool.", add_special_tokens=False).ids
    generations = [token, plain + [IM_END]] if order == "token first" else [plain + [IM_END], token]
    replies = [served(adapter, messages, ids, **options) for ids in generations]
    assert replies[0]["content"] == replies[1]["content"]
    for reply in replies:
        assert TOOL_CALL not in adapter.encode(follow(messages, reply, **options))


def test_two_different_kept_sets_make_the_text_ambiguous(root):
    adapter = ModelTokenizer(root)
    messages, options = [dict(role="user", content="Quote it.")], dict(enable_thinking=False)
    encode = lambda text: adapter.tokenizer.encode(text, add_special_tokens=False).ids
    plain = lambda text: adapter.content.text_tokenizer().encode(text, add_special_tokens=False).ids
    first = encode("A <tool_call>") + plain(" and B <tool_call>.") + [IM_END]  # the first copy is the added token
    second = plain("A <tool_call> and B") + encode(" <tool_call>.") + [IM_END]  # the second copy is the added token
    replies = [served(adapter, messages, ids, **options) for ids in (first, second, first)]
    assert len({reply["content"] for reply in replies}) == 1
    for reply in replies:
        assert TOOL_CALL not in adapter.encode(follow(messages, reply, **options))


def test_the_same_kept_set_twice_stays_kept(root):
    adapter = ModelTokenizer(root)
    generated, options, messages = CASES["stray token in content"]
    ids = adapter.tokenizer.encode(generated, add_special_tokens=False).ids
    replies = [served(adapter, messages, ids, **options) for _ in range(2)]
    assert TOOL_CALL in adapter.encode(follow(messages, replies[1], **options))
