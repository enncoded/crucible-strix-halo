"""A fake `python -m server` for tools/serve_parity.py tests: deterministic token IDs, no engine, no GPU.

The IDs depend only on the request fields that a real exact engine keeps (messages, seed, temperature, sampling), so
a serial and a `--streams 1` fake agree. `--streams` makes it a batch fake: it refuses the requests that the real batch
server refuses. FAKE_PARITY_DIVERGE=<name part> makes the batch fake differ on a case.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os

BATCH_REFUSED = (("draft_head", "full"), ("draft_adapt", 1), ("draft_q", 0))


def make_ids(body, diverge):
    seed = json.dumps([body["messages"], body.get("temperature"), body.get("top_p"), body.get("top_k"),
                       body.get("min_p"), body.get("seed"), body.get("max_tokens")], sort_keys=True)
    if diverge:
        seed += "|batch"
    digest = hashlib.sha256(seed.encode()).digest()
    count = min(int(body.get("max_tokens", 16)), 24)
    return [int.from_bytes(digest[index % 32:index % 32 + 2], "big") % 1000 for index in range(count)]


def has_image(body):
    return any(isinstance(message["content"], list) and any(part.get("type") == "image_url" for part in message["content"])
               for message in body["messages"])


class Handler(BaseHTTPRequestHandler):
    batch = False
    vision = False
    seen = set()

    def log_message(self, *_):
        pass

    def reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply(200, {"data": [{"id": "crucible"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.batch and any(body.get(key) == value for key, value in BATCH_REFUSED
                              if key in body and type(body[key]) is type(value)):
            return self.reply(400, {"error": {"message": "The batched server does not support this draft option. Use --serial (rollback only)"}})
        if has_image(body) and not self.vision:
            return self.reply(400, {"error": {"message": "start serve with --vision"}})
        marker = os.environ.get("FAKE_PARITY_DIVERGE", "")
        diverge = self.batch and bool(marker) and marker in json.dumps(body["messages"])
        ids = make_ids(body, diverge)
        key = json.dumps(body["messages"][:1])
        hit = key in self.seen and body.get("cache_control") != "bypass"
        type(self).seen = self.seen | {key}
        metrics = {"restored_tokens": 10 if hit else 0, "session_label": "hit_session" if hit else "miss"}
        self.reply(200, {"choices": [{"index": 0, "finish_reason": "length", "token_ids": ids,
                                      "message": {"role": "assistant", "content": " ".join(map(str, ids))}}],
                         "usage": {"prompt_tokens": 20, "completion_tokens": len(ids), "crucible_metrics": metrics}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--engine")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--streams", type=int)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--ctx")
    parser.add_argument("--prefill")
    parser.add_argument("--conversation-cache-mib")
    parser.add_argument("--session-replay")
    args = parser.parse_args()
    Handler.batch = not args.serial  # a missing --streams is --streams 1
    Handler.vision = args.vision
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
