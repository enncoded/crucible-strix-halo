"""In-memory HTTP transport and scripted token events. No listener is needed."""
import io
import json
import queue
import threading
from email.message import Message
from types import SimpleNamespace

from server.http import Service, handler_for
from server.metrics import Trace


class PiecesTokenizer:
    def __init__(self, pieces, *, tail="", decoded=None, prompt_ids=(10,)):
        self.pieces = dict(enumerate(pieces, 20))
        self.tail, self.decoded, self.prompt_ids = tail, decoded, prompt_ids

    def encode(self, request):
        return list(request.prompt) if isinstance(request.prompt, tuple) else list(self.prompt_ids)

    def stream(self):
        return None

    def step(self, stream, token):
        return self.pieces.get(token, "")

    def decode(self, ids):
        return self.decoded if self.decoded is not None else "".join(self.pieces.get(token, "") for token in ids) + self.tail


class ScriptBackend:
    def __init__(self, events):
        self.source, self.jobs, self.submissions = events, [], []
        self.trace = Trace()

    def submit(self, ids, request, **options):
        self.submissions.append((list(ids), request, options))
        events = queue.Queue()
        for event in self.source:
            events.put(dict(event))
        job = SimpleNamespace(events=events, cancelled=threading.Event())
        self.jobs.append(job)
        return job


def token_events(pieces, *, stop_token=True, **done):
    ids = list(range(20, 20 + len(pieces))) + ([248046] if stop_token else [])
    return [{"event": "token", "id": token} for token in ids] + [{
        "event": "done", "prompt_tokens": 1, "completion_tokens": len(ids),
        "finish_reason": "stop" if stop_token else "length", **done,
    }]


class DisconnectWriter(io.BytesIO):
    def __init__(self, fail_after, error):
        super().__init__()
        self.fail_after, self.error, self.writes = fail_after, error, 0

    def write(self, data):
        self.writes += 1
        if self.writes > self.fail_after:
            raise self.error("client disconnected")
        return super().write(data)


def invoke(backend, tokenizer, payload=None, *, path="/v1/chat/completions", method="POST", raw=None,
           headers=None, writer=None, service_options=None, connection=None):
    handler_class = handler_for(Service(backend, tokenizer, **(service_options or {})))
    handler = handler_class.__new__(handler_class)
    handler.path = path
    body = raw if raw is not None else json.dumps(payload).encode()
    handler.headers = Message()
    values = [("Host", "127.0.0.1:8790"), ("Content-Length", str(len(body))), ("Content-Type", "application/json")]
    if headers is not None:
        values = headers
    for name, value in values:
        handler.headers[name] = value
    handler.server = SimpleNamespace(server_address=("127.0.0.1", 8790))
    if connection is not None:
        handler.connection = connection
    handler.rfile, handler.wfile = io.BytesIO(body), writer if writer is not None else io.BytesIO()
    statuses, response_headers = [], {}
    handler.send_response = statuses.append
    handler.send_header = lambda name, value: response_headers.update({name: value})
    handler.end_headers = lambda: None
    getattr(handler, "do_" + method)()
    return SimpleNamespace(statuses=statuses, headers=response_headers, body=handler.wfile.getvalue(), handler=handler)


def chunks(response):
    return [json.loads(part[6:]) for part in response.body.decode().split("\n\n") if part.startswith("data: {")]


def joined_calls(deltas):
    """Streamed tool calls joined by index, as OpenAI clients join them."""
    calls = {}
    for call in deltas:
        entry = calls.setdefault(call["index"], dict(id=None, type=None, function=dict(name="", arguments="")))
        entry["id"] = entry["id"] or call.get("id")
        entry["type"] = entry["type"] or call.get("type")
        function = call.get("function", {})
        entry["function"]["name"] += function.get("name", "")
        entry["function"]["arguments"] += function.get("arguments", "")
    return [calls[index] for index in sorted(calls)]


def chat(**options):
    return {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16, "seed": 42, **options}
