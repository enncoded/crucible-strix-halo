"""Streaming, tools, stops, usage, and rejected requests through the HTTP handler."""
import json
import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from server.api import ApiError
from server.http import Service
from server.tests.serve_fixtures import (
    DisconnectWriter, PiecesTokenizer, ScriptBackend, chat, chunks, invoke, joined_calls, token_events,
)
from server.tests.test_tools import TOOLS, ToolOutputTests


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("split", range(8))
def test_stop_across_chunk_edges_has_exact_usage(stream, split):
    text = "prefix-STOP-tail"
    edge = len("prefix") + split
    pieces = [text[:edge], text[edge:]]
    backend = ScriptBackend(token_events(pieces))
    response = invoke(backend, PiecesTokenizer(pieces), chat(stop="-STOP-", stream=stream,
                      return_token_ids=True, stream_options={"include_usage": True}))
    assert response.statuses == [200]
    if stream:
        events = chunks(response)
        choices = [choice for event in events for choice in event["choices"]]
        assert "".join(choice["delta"].get("content", "") for choice in choices) == "prefix"
        ids = [choice["token_id"] for choice in choices if "token_id" in choice]
        assert choices[-1]["finish_reason"] == "stop"
        usage = events[-1]["usage"]
        assert all(event["usage"] is None for event in events[:-1])
        assert response.body.endswith(b"data: [DONE]\n\n")
    else:
        event = json.loads(response.body)
        choice = event["choices"][0]
        assert choice["message"]["content"] == "prefix"
        assert choice["finish_reason"] == "stop"
        ids, usage = choice["token_ids"], event["usage"]
    assert len(ids) == (1 if edge >= len("prefix-STOP-") else 2)
    assert usage == {"prompt_tokens": 1, "completion_tokens": len(ids), "total_tokens": 1 + len(ids)}
    assert backend.jobs[0].cancelled.is_set()


@pytest.mark.parametrize("stream", [False, True])
def test_parallel_tool_calls_use_distinct_ids_and_indices(stream):
    block = ToolOutputTests().block
    text = "Before.\n" + block("café.py") + "\n" + block("雪.py")
    pieces = list(text)
    backend = ScriptBackend(token_events(pieces))
    response = invoke(backend, PiecesTokenizer(pieces), chat(tools=TOOLS, max_tokens=len(pieces) + 1,
                      stream=stream, stream_options={"include_usage": True}))
    assert response.statuses == [200]
    if stream:
        events = chunks(response)
        choices = [choice for event in events for choice in event["choices"]]
        deltas = [call for choice in choices for call in choice["delta"].get("tool_calls", [])]
        assert sorted({call["index"] for call in deltas}) == [0, 1]
        # Calls stream in order: every delta of call 0 comes before call 1 starts.
        assert [call["index"] for call in deltas] == sorted(call["index"] for call in deltas)
        calls = joined_calls(deltas)
        assert "".join(choice["delta"].get("content", "") for choice in choices) == "Before.\n"
        assert choices[-1]["finish_reason"] == "tool_calls"
        assert events[-1]["usage"]["completion_tokens"] == len(pieces) + 1
        assert b"<tool_call>" not in response.body
    else:
        event = json.loads(response.body)
        calls = event["choices"][0]["message"]["tool_calls"]
        assert event["choices"][0]["finish_reason"] == "tool_calls"
        assert event["choices"][0]["message"]["content"] == "Before.\n"
        assert event["usage"]["completion_tokens"] == len(pieces) + 1
    assert len({call["id"] for call in calls}) == 2
    assert [json.loads(call["function"]["arguments"])["path"] for call in calls] == ["café.py", "雪.py"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("options,text,code,started", [
    ({"parallel_tool_calls": False}, ToolOutputTests().block() * 2, "parallel_calls", True),
    ({"tool_choice": "required"}, "No call.", "tool_choice", False),
    ({"tool_choice": "none"}, ToolOutputTests().block(), "tool_choice", False),
    ({}, "<tool_call><function=read><parameter=path>x</function></tool_call>", "incomplete_call", True),
    ({}, "<tool_call><function=read><parameter=path>x</parameter>junk</function></tool_call>", "parameter_tag", True),
    ({}, "<tool", "incomplete_call", False),
    ({}, ToolOutputTests().block() * 129, "too_many_calls", True),
])
def test_tool_policy_failures_never_publish_unvalidated_calls(stream, options, text, code, started):
    # The text comes back as content (design decision, Oct 6, as other engines); never a valid call, never an error.
    # A stream that already sent a call's deltas ends with "length", so a client does not run it (design decision, Oct 7).
    backend = ScriptBackend(token_events([text]))
    usage_option = {"stream_options": {"include_usage": True}} if stream else {}
    response = invoke(backend, PiecesTokenizer([text]), chat(tools=TOOLS, stream=stream, **options, **usage_option))
    assert response.statuses == [200]
    assert b'"error"' not in response.body
    assert (b'"tool_calls"' in response.body) == (stream and started)
    assert backend.jobs[0].cancelled.is_set()
    if stream:
        assert response.body.endswith(b"data: [DONE]\n\n")
        parts = chunks(response)
        assert "".join((part["choices"][0]["delta"].get("content") or "") for part in parts
                       if part.get("choices")) == text
        assert parts[-2]["choices"][0]["finish_reason"] == ("length" if started else "stop")
        usage = parts[-1]["usage"]
    else:
        body = json.loads(response.body)
        assert body["choices"][0]["message"]["content"] == text
        assert body["choices"][0]["finish_reason"] == "stop"
        usage = body["usage"]
    assert usage["tool_call_fallback"] is True and usage["tool_call_invalid"] == code


@pytest.mark.parametrize("stream", [False, True])
def test_decode_tail_flushes_partial_stop_and_keeps_cache_usage(stream):
    pieces = ["hello EN"]
    backend = ScriptBackend(token_events(pieces, cached_tokens=1))
    response = invoke(backend, PiecesTokenizer(pieces, tail="x"), chat(stop="END", stream=stream,
                      stream_options={"include_usage": True}))
    event = chunks(response)[-1] if stream else json.loads(response.body)
    assert event["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3,
                             "prompt_tokens_details": {"cached_tokens": 1}}
    if stream:
        assert "".join(choice["delta"].get("content", "") for item in chunks(response) for choice in item["choices"]) == "hello ENx"
    else:
        assert event["choices"][0]["message"]["content"] == "hello ENx"


@pytest.mark.parametrize("stream", [False, True])
def test_inconsistent_decode_is_explicit_and_cancels_job(stream):
    backend = ScriptBackend(token_events(["hello"]))
    response = invoke(backend, PiecesTokenizer(["hello"], decoded="other"), chat(stream=stream))
    assert response.statuses == ([200] if stream else [500])
    assert b"incremental tokenizer output is inconsistent" in response.body
    assert backend.jobs[0].cancelled.is_set()


@pytest.mark.parametrize("position", ["before", "during"])
def test_engine_errors_before_and_after_stream_headers(position):
    error = {"event": "error", "error": "fixture failure"}
    events = [error] if position == "before" else [{"event": "token", "id": 20}, error]
    backend = ScriptBackend(events)
    response = invoke(backend, PiecesTokenizer(["hello"]), chat(stream=True))
    # A stream sends its headers at admission, so an error before the first token also ends it as an SSE error.
    assert response.statuses == [200]
    assert b"fixture failure" in response.body
    assert backend.jobs[0].cancelled.is_set()
    assert response.body.endswith(b"data: [DONE]\n\n")


def test_engine_error_before_output_of_a_plain_request_is_503():
    backend = ScriptBackend([{"event": "error", "error": "fixture failure"}])
    response = invoke(backend, PiecesTokenizer(["hello"]), chat())
    assert response.statuses == [503]
    assert b"fixture failure" in response.body


@pytest.mark.parametrize("error", [BrokenPipeError, ConnectionResetError, TimeoutError])
@pytest.mark.parametrize("stream", [False, True])
def test_client_write_failure_always_cancels_job(error, stream):
    backend = ScriptBackend(token_events(["hello", " world"]))
    writer = DisconnectWriter(2 if stream else 0, error)
    response = invoke(backend, PiecesTokenizer(["hello", " world"]), chat(stream=stream), writer=writer)
    assert response.handler.close_connection
    assert backend.jobs[0].cancelled.is_set()


@pytest.mark.parametrize("event", [
    {"event": "token", "id": True}, {"event": "token", "id": -1},
    {"event": "token", "id": 248320}, {"event": "token"}, {"event": "unexpected"},
    {"event": "done", "prompt_tokens": 9, "completion_tokens": 0, "finish_reason": "length"},
])
def test_invalid_engine_events_return_a_server_error(event):
    backend = ScriptBackend([event])
    response = invoke(backend, PiecesTokenizer([]), chat())
    assert response.statuses == [500]
    assert json.loads(response.body)["error"]["type"] == "server_error"
    assert backend.jobs[0].cancelled.is_set()


def test_queued_event_timeout_returns_503():
    job = SimpleNamespace(events=Mock())
    job.events.get.side_effect = queue.Empty
    with pytest.raises(ApiError) as error: Service.next_event(job)
    assert error.value.status == 503
    job.events.get.assert_called_once_with(timeout=360)


@pytest.mark.parametrize("body,param", [
    ([], None), (chat(stream=1), "stream"), (chat(model="missing"), "model"),
    (chat(cache_control="purge"), "cache_control"), (chat(draft_head="bad"), "draft_head"),
    (chat(draft_adapt=1, draft_head="full"), "draft_adapt"), (chat(reasoning_effort="extreme"), "reasoning_effort"),
    (chat(stream_options={"unknown": True}), "stream_options"), (chat(prompt="bad"), "prompt"),
    (chat(max_completion_tokens=1), "max_tokens"), (chat(messages=[{"role": "user", "content": 4}]), "messages"),
    (chat(tool_choice="required"), "tool_choice"), (chat(tools={}), "tools"),
    (chat(tools=[{"type": "invalid"}]), "tools"),
    (chat(tools=[{"type": "function", "function": {"name": "f", "strict": 1}}]), "tools"),
    (chat(tools=[{"type": "function", "function": {"name": "f", "description": 1}}]), "tools"),
])
def test_validation_rejects_before_backend_submission(body, param):
    backend = ScriptBackend([])
    response = invoke(backend, PiecesTokenizer([]), body)
    assert response.statuses == ([404] if param == "model" else [400])
    assert json.loads(response.body)["error"]["param"] == param
    assert not backend.submissions


@pytest.mark.parametrize("headers,raw,status", [
    ([("Transfer-Encoding", "chunked")], b"{}", 400),
    ([("Content-Length", "2"), ("Content-Length", "2")], b"{}", 411),
    ([], b"{}", 411), ([("Content-Length", "0")], b"", 400),  # SH-1: empty is not too large
    ([("Content-Length", "1048577")], b"{}", 413),
    ([("Content-Length", "2"), ("Content-Type", "text/plain")], b"{}", 415),
    ([("Content-Length", "10"), ("Content-Type", "application/json")], b"{}", 400),
    ([("Content-Length", "1"), ("Content-Type", "application/json")], b"\xff", 400),
    ([("Content-Length", "3"), ("Content-Type", "application/json")], b"NaN", 400),
])
def test_transport_errors_do_not_submit(headers, raw, status):
    backend = ScriptBackend([])
    response = invoke(backend, PiecesTokenizer([]), raw=raw, headers=[("Host", "127.0.0.1:8790")] + headers)
    assert response.statuses == [status]
    assert not backend.jobs


def test_body_limit_follows_max_body_mib():
    body = json.dumps(dict(model="crucible", messages=[dict(role="user", content="x" * (2 * 1024 * 1024))],
                           max_tokens=8)).encode()
    for options, status in ((None, 413), (dict(max_body_mib=4), 200)):
        backend = ScriptBackend(token_events(["ok"]) if status == 200 else [])
        response = invoke(backend, PiecesTokenizer(["ok"]), raw=body, service_options=options)
        assert response.statuses == [status]
        if status == 413:
            assert b"1 MiB" in response.body and not backend.jobs


def test_models_trace_and_unknown_endpoints():
    backend, tokenizer = ScriptBackend([]), PiecesTokenizer([])
    response = invoke(backend, tokenizer, method="GET", path="/v1/models")
    assert response.statuses == [200]
    assert json.loads(response.body)["data"][0]["id"] == "crucible"
    backend.trace.add(1, 2, "step", [3])
    response = invoke(backend, tokenizer, method="GET", path="/v1/crucible/trace?since=1&since=2")
    assert json.loads(response.body)["data"][0]["request_ids"] == [3]
    for method, path, status in (("GET", "/missing", 404), ("POST", "/missing", 404),
                                 ("GET", "/v1/crucible/trace?since=bad", 400)):
        assert invoke(backend, tokenizer, method=method, path=path).statuses == [status]


def test_cache_frontiers_reach_backend_without_changing_prompt_ids():
    tokenizer = PiecesTokenizer(["hello"])
    tokenizer.cache_prefix_tokens = lambda request, ids: 1
    tokenizer.shared_prefix_tokens = lambda request, ids: 1
    backend = ScriptBackend(token_events(["hello"]))
    assert invoke(backend, tokenizer, chat()).statuses == [200]
    assert backend.submissions[0][0] == [10]
    assert backend.submissions[0][2] == {"cache_prefix_tokens": 1, "shared_prefix_tokens": 1}
