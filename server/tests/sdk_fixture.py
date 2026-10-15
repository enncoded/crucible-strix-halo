"""CPU-only recorded token scripts for real HTTP client compatibility tests."""
from collections import deque
from dataclasses import dataclass
import queue
import threading
from types import SimpleNamespace
from server.api import ApiError

@dataclass(frozen=True)
class Script:
    pieces: tuple = ("Hello", " snow 雪.")
    reason: str = "stop"
    error: object = None
    cached_tokens: int | None = None

class RecordedTokenizer:
    def __init__(self):
        self.pieces = {}
        self.next_id = 100

    def encode(self, request):
        return [10, 11, 12]

    def tokens(self, pieces):
        ids = tuple(range(self.next_id, self.next_id + len(pieces)))
        self.next_id += len(pieces)
        self.pieces.update(zip(ids, pieces))
        return ids

    def stream(self):
        return None

    def step(self, stream, token):
        return self.pieces.get(token, "")

    def decode(self, ids):
        return "".join(self.step(None, token) for token in ids)

class RecordedBackend:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.scripts = deque()
        self.requests = []
        self.jobs = []

    def reset(self, *scripts):
        self.scripts = deque(scripts)
        self.requests = []
        self.jobs = []

    def submit(self, ids, request):
        script = self.scripts.popleft() if self.scripts else Script()
        if isinstance(script.error, ApiError):
            raise script.error
        self.requests.append(request)
        events = queue.Queue()
        if script.error:
            events.put(dict(event="error", error=script.error))
        else:
            tokens = self.tokenizer.tokens(script.pieces)
            limited = len(tokens) > request.max_tokens
            tokens = tokens[:request.max_tokens]
            for token in tokens:
                events.put(dict(event="token", id=token))
            cached = {} if script.cached_tokens is None else dict(cached_tokens=script.cached_tokens)
            events.put(dict(event="done", prompt_tokens=len(ids), completion_tokens=len(tokens),
                            finish_reason="length" if limited else script.reason) | cached)
        job = SimpleNamespace(events=events, cancelled=threading.Event())
        self.jobs.append(job)
        return job

TOOLS = [dict(type="function", function=dict(name="add", parameters=dict(type="object",
    properties=dict(a=dict(type="integer"), b=dict(type="integer")), required=["a", "b"],
    additionalProperties=False)))]

def tool_block(a=17, b=25):
    return (f"<tool_call>\n<function=add>\n<parameter=a>\n{a}\n</parameter>\n"
            f"<parameter=b>\n{b}\n</parameter>\n</function>\n</tool_call>")
