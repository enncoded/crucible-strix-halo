"""Every 4xx has a crucible_error line with a fixed reason code and counts only (train-13 h2h: two context-full
400s could be seen only from token counts). The client text does not change, and request text never reaches the log."""
import json
import pytest
from server.api import ApiError
from server.metrics import CLIENT_ERROR_REASONS
from server.tests.serve_fixtures import PiecesTokenizer, ScriptBackend, chat, invoke, token_events
from server.tests.test_backend_paths import request, start_backend

CANARY = "CANARY-client-error-4471"


def error_lines(capsys):
    return [json.loads(line.split(" ", 1)[1]) for line in capsys.readouterr().err.splitlines()
            if line.startswith("crucible_error ")]


class RaisingBackend(ScriptBackend):
    def __init__(self, error):
        super().__init__(token_events(["a"]))
        self.error = error

    def submit(self, ids, request, **options):
        raise self.error


def test_context_full_400_logs_its_reason_and_counts(start_backend, capsys):
    backend = start_backend(context=32)
    with pytest.raises(ApiError) as caught:
        backend.submit(list(range(1, 30)), request(8))
    assert caught.value.code == "context_length"
    assert caught.value.counts == dict(prompt_tokens=29, max_tokens=8, context=32)
    response = invoke(RaisingBackend(caught.value), PiecesTokenizer(["a"]), chat(messages=[dict(role="user", content=CANARY)]))
    assert response.statuses == [400] and "exceeds the context window" in json.loads(response.body)["error"]["message"]
    lines = error_lines(capsys)
    assert lines == [dict(request_id=None, status=400, streamed=False, reason="context_length",
                          prompt_tokens=29, max_tokens=8, context=32)]


@pytest.mark.parametrize("body,status,reason", [
    (chat(messages=[dict(role="user", content=CANARY)], temperature="hot"), 400, "bad_request"),
    (chat(messages=[dict(role="user", content="x"), dict(role="assistant", content=None, tool_calls=[dict(
        id="c1", type="function", function=dict(name="read", arguments='{"' + CANARY + '": [1'))])],
        tools=[dict(type="function", function=dict(name="read", parameters=dict(type="object")))]),
     400, "invalid_tool_call_history"),
])
def test_validation_4xx_logs_a_fixed_reason_and_no_request_text(capsys, body, status, reason):
    response = invoke(ScriptBackend(token_events(["a"])), PiecesTokenizer(["a"]), body)
    assert response.statuses == [status]
    err = capsys.readouterr().err
    assert CANARY not in err
    lines = [json.loads(line.split(" ", 1)[1]) for line in err.splitlines() if line.startswith("crucible_error ")]
    assert lines == [dict(request_id=None, status=status, streamed=False, reason=reason)]


def test_engine_line_413_and_unknown_path_have_reasons(start_backend, capsys):
    from unittest.mock import patch
    from server import backend as backend_module
    backend = start_backend(context=4096, env={"FAKE_CONTEXT": "4096"})
    with patch.object(backend, "engine_line_bytes", 200), pytest.raises(ApiError) as caught:
        backend.submit(list(range(1, 100)), request())
    assert (caught.value.status, caught.value.code) == (413, "line_too_large")
    response = invoke(ScriptBackend([]), PiecesTokenizer([]), path="/v1/" + CANARY, method="GET")
    assert response.statuses == [404]
    lines = error_lines(capsys)
    assert lines == [dict(request_id=None, status=404, streamed=False, reason="not_found")]


def test_every_reason_is_from_the_closed_set():
    from server.metrics import client_error_line
    line = json.loads(client_error_line(None, 400, ApiError(CANARY + " text", code=CANARY)).split(" ", 1)[1])
    assert line["reason"] == "bad_request" and CANARY not in json.dumps(line)
    assert {"context_length", "invalid_tool_call_history", "line_too_large", "bad_request", "not_found"} <= CLIENT_ERROR_REASONS
