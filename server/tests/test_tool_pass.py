"""A complete, well-formed tool call that fails only the schema goes out as a tool call (design decision, Oct 8).

The client checks the call against its own schema and tells the model the exact error. Before, the server returned the
raw XML as content with finish "length", so Pi said the output was truncated and the model repeated the bad parameter.
A malformed or incomplete call keeps the old fallback.
"""
import json

import pytest

from server.tests.serve_fixtures import PiecesTokenizer, ScriptBackend, chat, chunks, invoke, joined_calls, token_events

SHELL = dict(type="function", function=dict(name="shell_job_start", description="Start a shell job.", parameters=dict(
    type="object", additionalProperties=False, required=["command"],
    properties=dict(command=dict(type="string"), cwd=dict(type="string"), title=dict(type="string"),
                    lines=dict(type="integer"), mode=dict(type="string", enum=["fg", "bg"])))))
STRICT = dict(type="function", function=dict(SHELL["function"], name="strict_job", strict=True))


def call(name, **parameters):
    body = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in parameters.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def serve(text, stream, stream_tool_calls=True, tools=(SHELL,), **options):
    """(finish reason, content, calls, usage, log lines) for one model output through the HTTP layer."""
    import contextlib, io
    backend = ScriptBackend(token_events([text]))
    body = chat(tools=list(tools), stream=stream, **options) | (
        dict(stream_options=dict(include_usage=True)) if stream else {})
    log = io.StringIO()
    with contextlib.redirect_stderr(log):
        response = invoke(backend, PiecesTokenizer([text]), body,
                          service_options=dict(stream_tool_calls=stream_tool_calls))
    lines = [line for line in log.getvalue().splitlines() if line.startswith("crucible_")]
    assert response.statuses == [200]
    if stream:
        assert response.body.endswith(b"data: [DONE]\n\n")
        parts = chunks(response)
        choices = [part["choices"][0] for part in parts if part.get("choices")]
        content = "".join(choice["delta"].get("content") or "" for choice in choices)
        calls = joined_calls([delta for choice in choices for delta in choice["delta"].get("tool_calls", [])])
        return choices[-1]["finish_reason"], content, calls, parts[-1]["usage"], lines
    reply = json.loads(response.body)
    message = reply["choices"][0]["message"]
    return reply["choices"][0]["finish_reason"], message["content"] or "", message.get("tool_calls", []), \
        reply["usage"], lines


PASS = [
    # The repro (serve-t14b, Oct 8): an unknown parameter on a tool with additionalProperties false.
    (call("shell_job_start", command="ls -la", timeoutMinutes="5"), dict(command="ls -la", timeoutMinutes=5),
     "schema_mismatch"),
    (call("shell_job_start", cwd="/tmp"), dict(cwd="/tmp"), "schema_mismatch"),                       # missing required
    (call("shell_job_start", command="ls", mode="up"), dict(command="ls", mode="up"), "schema_mismatch"),  # enum
    (call("shell_job_start", command="ls", lines="many"), dict(command="ls", lines="many"), "parameter_json"),
    (call("shell_job_start", command="ls", lines="[1]"), dict(command="ls", lines=[1]), "schema_mismatch"),
    (call("job_stop", id="7"), dict(id=7), "unknown_function"),
]


@pytest.mark.parametrize("raw,arguments,code", PASS)
@pytest.mark.parametrize("stream,stream_tool_calls", [(False, True), (True, True), (True, False)])
def test_a_well_formed_call_that_fails_the_schema_goes_out_as_a_call(raw, arguments, code, stream, stream_tool_calls):
    text = "I will run it.\n" + raw
    reason, content, calls, usage, lines = serve(text, stream, stream_tool_calls)
    assert reason == "tool_calls"
    assert content == "I will run it.\n" and "<tool_call>" not in content
    assert [json.loads(c["function"]["arguments"]) for c in calls] == [arguments]
    assert calls[0]["function"]["name"] == raw.split("<function=")[1].split(">")[0]
    assert "tool_call_fallback" not in usage
    assert not [line for line in lines if line.startswith("crucible_error ")]
    passes = [json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith("crucible_tool_pass ")]
    assert passes == [dict(request_id=None, streamed=stream, calls=1, reasons=[code])]
    # The streamed arguments are byte-equal to the plain reply's arguments.
    plain = serve(text, False)[2]
    assert [c["function"]["arguments"] for c in calls] == [c["function"]["arguments"] for c in plain]


@pytest.mark.parametrize("raw,code", [
    ("<tool_call>\n<function=shell_job_start>\n<parameter=command>\nls", "incomplete_call"),
    (call("shell_job_start", command="ls") + " trailing", "unexpected_text"),
    ("<tool_call><function=shell_job_start><parameter=command>x</parameter>junk</function></tool_call>",
     "parameter_tag"),
    ("<tool_call>\n<function=shell_job_start>\n<parameter=command>\na\n</parameter>\n<parameter=command>\nb\n"
     "</parameter>\n</function>\n</tool_call>", "duplicate_parameter"),
    (call("bad name", command="ls"), "unknown_function"),  # a name that the next request could not send back
    (call("strict_job", command="ls", timeoutMinutes="5"), "schema_mismatch"),  # strict: true keeps the old rule
])
@pytest.mark.parametrize("stream", [False, True])
def test_malformed_incomplete_and_strict_calls_keep_the_fallback(raw, code, stream):
    text = "I will run it.\n" + raw
    reason, content, calls, usage, lines = serve(text, stream, tools=(SHELL, STRICT))
    assert content == text
    assert usage["tool_call_fallback"] is True and usage["tool_call_invalid"] == code
    # A stream that already sent a call's header ends with "length", so the client does not run the partial call.
    assert reason == ("length" if calls else "stop") and (not calls or stream)
    assert not [line for line in lines if line.startswith("crucible_tool_pass ")]


def test_a_streamed_header_then_a_malformed_call_keeps_length():
    text = "<tool_call>\n<function=shell_job_start>\n<parameter=command>\nls\n</parameter>\njunk</function>\n</tool_call>"
    reason, content, calls, usage, _ = serve(text, True)
    assert (reason, content, usage["tool_call_invalid"]) == ("length", text, "parameter_tag")
    assert calls and calls[0]["function"]["arguments"] == '{"command":"ls"'  # never a complete object


def test_policy_failures_keep_the_fallback():
    raw = call("shell_job_start", command="ls", extra="1")
    for options, code in ((dict(tool_choice="none"), "tool_choice"),
                          (dict(parallel_tool_calls=False), "parallel_calls"),
                          (dict(tool_choice=dict(type="function", function=dict(name="strict_job"))), "tool_choice")):
        text = raw + ("\n" + raw if code == "parallel_calls" else "")
        for stream in (False, True):
            reason, content, calls, usage, _ = serve(text, stream, tools=(SHELL, STRICT), **options)
            assert (content, usage["tool_call_invalid"]) == (text, code)
            assert reason == ("length" if calls else "stop")


def test_two_calls_with_one_unchecked_report_one_reason():
    text = call("shell_job_start", command="ls") + "\n" + call("shell_job_start", command="pwd", lines="x")
    for stream in (False, True):
        reason, _, calls, _, lines = serve(text, stream)
        assert reason == "tool_calls" and len(calls) == 2
        passes = [json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith("crucible_tool_pass ")]
        assert passes == [dict(request_id=None, streamed=stream, calls=2, reasons=["parameter_json"])]


@pytest.mark.parametrize("stream", [False, True])
def test_a_valid_call_writes_no_pass_line(stream):
    # a mutant that wrote the line for every call survived the tests above.
    reason, _, calls, usage, lines = serve(call("shell_job_start", command="ls", lines="3"), stream)
    assert (reason, len(calls)) == ("tool_calls", 1) and "tool_call_fallback" not in usage
    assert not [line for line in lines if line.startswith("crucible_")]


def test_required_choice_with_an_unknown_function_goes_out_as_a_call():
    # with tool_choice "required", an unknown valid-pattern name counts as the required call.
    for stream in (False, True):
        reason, _, calls, _, lines = serve(call("job_stop", id="7"), stream, tool_choice="required")
        assert (reason, calls[0]["function"]["name"]) == ("tool_calls", "job_stop")
        assert any(line.startswith("crucible_tool_pass ") for line in lines)


@pytest.mark.parametrize("depth", range(60, 70))
def test_a_generated_value_is_accepted_back_in_the_next_request(depth):
    # a value with 65 levels passed, but the whole arguments object then had 66, and the next request
    # got HTTP 400 ("JSON nesting must not exceed 64 levels"). Both sides now count from the arguments object.
    from server.api import validate_request
    value = "[" * depth + "]" * depth
    for stream in (False, True):
        _, _, calls, _, _ = serve(call("shell_job_start", command="ls", nested=value), stream)
        history = [dict(role="user", content="Run it."),
                   dict(role="assistant", content=None, tool_calls=[dict(calls[0], type="function")]),
                   dict(role="tool", tool_call_id=calls[0]["id"], content="ok")]
        validate_request(dict(messages=history, tools=[SHELL]), "chat", "crucible")
