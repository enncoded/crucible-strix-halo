"""Run under gpu-run. Check real tool calls and exact greedy input/token parity."""
import argparse
from datetime import datetime
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import socket
import subprocess
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from server.api import STOP_IDS, validate_request
from server.backend import EngineBackend
from server.http import Service, handler_for
from server.tokenization import ModelTokenizer

ROOT = Path(__file__).resolve().parents[2]
URL = "http://127.0.0.1:8790/v1/chat/completions"
TOOLS = [dict(type="function", function=dict(name="add", description="Add two integers.", parameters=dict(
    type="object", properties=dict(a=dict(type="integer"), b=dict(type="integer")), required=["a", "b"], additionalProperties=False)))]

def post(out, name, body):
    request = Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(request, timeout=90) as response:
        raw = response.read()
    (out / (name + ".response")).write_bytes(raw)
    if not body.get("stream"):
        return json.loads(raw)
    events = [json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith("data: {")]
    assert raw.endswith(b"data: [DONE]\n\n") and not any("error" in event for event in events), events
    return events

def greedy_ids(engine, checkpoint, tokenizer, body, out):
    ids = tokenizer.encode(validate_request(body, "chat", "crucible"))
    path = out / "tool-prompt.ids"
    path.write_text(",".join(map(str, ids)))
    result = subprocess.run([str(engine), "gen", str(checkpoint), "--ids-file", str(path), "--n", str(body["max_tokens"]),
                             "--spec", "2", "--temperature", "0"], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=True, timeout=120, text=True)
    (out / "gen.log").write_text(result.stdout)
    line = next(line for line in result.stdout.splitlines() if line.startswith("ids:"))
    generated = [int(value) for value in line[4:].split()]
    for index, value in enumerate(generated):
        if value in STOP_IDS:
            return generated[:index + 1]
    return generated

def tool_roundtrip(out, body, expected):
    plain = post(out, "tool-call", body)
    choice = plain["choices"][0]
    assert choice["token_ids"] == expected and choice["finish_reason"] == "tool_calls", plain
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "add" and json.loads(call["function"]["arguments"]) == dict(a=17, b=25), call
    assert plain["id"].startswith("chatcmpl-") and call["id"].startswith("call_"), plain
    events = post(out, "tool-stream", body | dict(stream=True, stream_options=dict(include_usage=True)))
    choices = [choice for event in events for choice in event["choices"]]
    assert [choice["token_id"] for choice in choices if "token_id" in choice] == expected
    deltas = [call for choice in choices for call in choice["delta"].get("tool_calls", [])]
    assert len(deltas) == 1 and deltas[0]["index"] == 0
    assert deltas[0]["id"].startswith("call_") and deltas[0]["id"] != call["id"]
    assert len({event["id"] for event in events}) == 1 and events[0]["id"] != plain["id"]
    assert deltas[0]["function"] == call["function"] and choices[-1]["finish_reason"] == "tool_calls"
    message = choice["message"]
    follow = body | dict(messages=body["messages"] + [message, dict(role="tool", tool_call_id=call["id"], content="42"),
                                                     dict(role="user", content="Reply with only the integer result from the tool.")],
                        tool_choice="none", max_tokens=32)
    response = post(out, "tool-result", follow)
    assert "42" in response["choices"][0]["message"]["content"] and "tool_calls" not in response["choices"][0]["message"]
    print("PASS: real tool call, OpenAI streaming deltas, multi-turn result, and exact greedy gen IDs", flush=True)

def reject_schema_limits(out, body):
    for index, schema in enumerate((dict(type="object", description="x" * 131073),
                                   dict(type="object", **{"$defs": {"loop": {"$ref": "#/$defs/loop"}}},
                                        allOf=[{"$ref": "#/$defs/loop"}]))):
        tools = [dict(type="function", function=dict(name="add", parameters=schema))]
        request = Request(URL, json.dumps(body | dict(tools=tools)).encode(), {"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=10):
                raise AssertionError("over-limit schema was accepted")
        except HTTPError as error:
            raw = error.read()
            (out / ("schema-limit-" + str(index) + ".response")).write_bytes(raw)
            assert error.code == 400 and json.loads(raw)["error"]["param"] == "tools", raw
    print("PASS: schema size and cycle limits return HTTP 400 before generation", flush=True)


def seed_replay(out):
    body = dict(messages=[dict(role="user", content="List four color names, separated by commas.")],
                max_tokens=24, spec=0, temperature=0.7, top_k=50, top_p=0.95, return_token_ids=True)
    first = post(out, "random-seed", body)
    replay = post(out, "reported-seed", body | dict(seed=first["seed"]))
    assert 0 <= first["seed"] < 1 << 64 and first["choices"][0]["token_ids"] == replay["choices"][0]["token_ids"]
    print("PASS: omitted HTTP seed is reported and repeats when supplied explicitly", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--engine", type=Path, default=ROOT / "build/engine/crucible")
    args = parser.parse_args()
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", 8790)) != 0, "Port 8790 already has a listener"
    out = ROOT / "results" / ("tools-live-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True)
    tokenizer = ModelTokenizer(args.checkpoint)
    body = dict(messages=[dict(role="user", content="Use add to sum 17 and 25. Make exactly one call with a=17 and b=25.")],
                tools=TOOLS, tool_choice=dict(type="function", function=dict(name="add")),
                parallel_tool_calls=False, max_tokens=128, spec=2, temperature=0, seed=0, return_token_ids=True)
    expected = greedy_ids(args.engine, args.checkpoint, tokenizer, body, out)
    backend = EngineBackend(args.engine, args.checkpoint)
    server = None
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 8790), handler_for(Service(backend, tokenizer)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        reject_schema_limits(out, body)
        tool_roundtrip(out, body, expected)
        seed_replay(out)
        print("results: " + str(out), flush=True)
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        backend.close()
    assert backend.process.poll() is not None
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", 8790)) != 0
    print("PASS: temporary tool server and resident engine stopped", flush=True)

if __name__ == "__main__": main()
