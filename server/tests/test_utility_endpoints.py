"""POST /tokenize, /v1/tokenize (OpenAI-compatible shape) and /apply-template: the chat path's prompt, with no engine work."""
import json
import threading
import time

import pytest

from server.api import validate_request
from server.http import Service
from server.tests.test_count_tokens import CANARY, Server, adapter, corpus  # noqa: F401 (the adapter fixture)


def post(server, path, body):
    status, data = server.post(path, body)
    return status, json.loads(data)


@pytest.mark.parametrize("path", ["/tokenize", "/v1/tokenize"])
def test_tokenize_messages_equals_the_chat_prefill(adapter, path, capsys):
    server = Server(adapter, default_thinking=True)
    try:
        for name, _, openai in corpus():
            status, body = post(server, path, openai)
            assert status == 200, body
            assert len(body["tokens"]) == server.prompt_tokens(openai), name
            assert body["tokens"] == adapter.encode(validate_request(openai, "chat", "crucible", default_thinking=True))
    finally:
        server.close()
    assert CANARY not in capsys.readouterr().err


def test_tokenize_content_is_the_completion_prompt(adapter):
    server = Server(adapter)
    try:
        status, body = post(server, "/tokenize", dict(content="Hello <|im_end|> world", add_special=False))
        expected = adapter.encode(validate_request(dict(prompt="Hello <|im_end|> world"), "completion", "crucible"))
        assert (status, body) == (200, dict(tokens=expected))
        assert post(server, "/tokenize", dict(content=[1, 2, 3]))[1] == dict(tokens=[1, 2, 3])
        assert post(server, "/tokenize", dict(content=""))[1] == dict(tokens=[])
    finally:
        server.close()


def test_apply_template_returns_the_rendered_prompt(adapter):
    server = Server(adapter, default_thinking=True)
    try:
        for name, _, openai in corpus():
            status, body = post(server, "/apply-template", openai)
            assert status == 200, body
            request = validate_request(openai, "chat", "crucible", default_thinking=True)
            # RA-8: the render carries sentinels for content special-token strings; the client gets them back.
            render = getattr(adapter, "render_prompt", None)
            text, restore = render(request) if render else (adapter.render_chat(request), {})
            assert body["prompt"] == "".join(restore.get(c, c) for c in text), name
    finally:
        server.close()


def test_apply_template_restores_content_special_strings():
    # With RA-8 (train 15) the render carries sentinels; the client must get the text the model reads.
    class Sentinel:
        def render_prompt(self, request, add_generation_prompt=True):
            return "a\U00100000b", {"\U00100000": "<|im_end|>"}
    service = Service(None, Sentinel())
    request = validate_request(dict(messages=[dict(role="user", content="x")]), "chat", "crucible")
    assert service.prompt_text(request) == "a<|im_end|>b"


@pytest.mark.parametrize("path, body, status, param", [
    ("/tokenize", dict(content="x", with_pieces=True), 400, "with_pieces"),
    ("/tokenize", dict(content="x", messages=[dict(role="user", content="x")]), 400, "content"),
    ("/tokenize", dict(content=5), 400, "content"),
    ("/tokenize", dict(), 400, "content"),
    ("/tokenize", dict(messages=[dict(role="user", content="x")], unknown=1), 400, "unknown"),
    ("/v1/tokenize", dict(model="other", messages=[dict(role="user", content="x")]), 404, "model"),
    ("/apply-template", dict(messages=[]), 400, "messages"),
    ("/apply-template", dict(prompt="x"), 400, "messages"),
    ("/apply-template", [], 400, None),
])
def test_utility_errors_use_the_server_error_object(adapter, path, body, status, param, capsys):
    server = Server(adapter)
    try:
        got, data = post(server, path, body)
    finally:
        server.close()
    assert (got, data["error"]["param"]) == (status, param)
    lines = [json.loads(line.split(" ", 1)[1]) for line in capsys.readouterr().err.splitlines()
             if line.startswith("crucible_error ")]
    assert [line["status"] for line in lines] == [status]


def test_utility_calls_never_reach_the_engine(adapter, tmp_path):
    log = tmp_path / "engine.jsonl"
    server = Server(adapter, False, dict(FAKE_REQUEST_LOG=str(log), FAKE_DELAY="0.01"))
    chat = dict(messages=[dict(role="user", content="x" * 20000)])
    try:
        replies = []
        generation = threading.Thread(target=lambda: replies.append(server.post("/v1/chat/completions", dict(
            messages=[dict(role="user", content="hi")], max_tokens=200, spec=0))))
        generation.start()
        time.sleep(0.5)
        started = time.monotonic()
        for path, body in (("/tokenize", chat), ("/v1/tokenize", dict(content="x" * 20000)), ("/apply-template", chat)):
            assert server.post(path, body)[0] == 200
        seconds = time.monotonic() - started
        assert generation.is_alive()
        generation.join(timeout=30)
    finally:
        server.close()
    assert replies[0][0] == 200 and seconds < 1.0
    assert len(log.read_text().splitlines()) == 1


def test_tokenize_refuses_content_and_messages_together(adapter):
    # the request is ambiguous. (validate_request already refused "content" as an unknown chat field,
    # but its message named sampling fields; this one says what to change.)
    server = Server(adapter)
    try:
        status, body = post(server, "/tokenize", dict(content="x", messages=[dict(role="user", content="x")]))
    finally:
        server.close()
    assert (status, body["error"]["param"]) == (400, "content")
    assert body["error"]["message"] == "Send content or messages, not both."
