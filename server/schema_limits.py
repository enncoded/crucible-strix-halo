"""Bound schema work and keep caches local to one request or model output."""
import json

MAX_SCHEMA_WORK = 16384
MAX_SCHEMA_DEPTH = 64
MAX_SCHEMA_NODES = 4096
MAX_SCHEMA_BYTES = 128 * 1024
MAX_SCHEMA_REFERENCES = 512
MAX_REFERENCE_CHAIN = 32


class WorkBudget:
    def __init__(self):
        self.remaining = MAX_SCHEMA_WORK
        self.nodes = self.bytes = self.references = self.instance = 0
        self.caches = {"schema": {}, "types": {}, "matches": {}}
        self.active = set()
        self.frames = []

    def visit(self, depth=0):
        if depth > MAX_SCHEMA_DEPTH:
            raise ValueError("schema nesting exceeds 64 levels")
        self.remaining -= 1
        if self.remaining < 0:
            raise ValueError("schema work budget exceeded")
        for frame in self.frames:
            frame[1] = max(frame[1], depth)

    def memo(self, kind, key, depth, compute):
        self.visit(depth)
        cache = self.caches[kind]
        if key in cache:
            result, height = cache[key]
            self.visit(depth + height)
            return result
        active = (kind, key)
        if active in self.active:
            raise ValueError("cyclic schema references are not supported")
        self.active.add(active)
        frame = [depth, depth]
        self.frames.append(frame)
        try:
            result = compute()
            cache[key] = (result, frame[1] - depth)
            return result
        finally:
            self.frames.pop()
            self.active.remove(active)

    def new_instance(self):
        # Separate parser calls can reuse Python scalar identities. Use a scope and path instead.
        self.instance += 1
        return self.instance

    def inspect(self, schema):
        pending = [(schema, 0)]
        while pending:
            value, depth = pending.pop()
            self.visit(depth)
            self.nodes += 1
            if self.nodes > MAX_SCHEMA_NODES:
                raise ValueError("schema node limit exceeded")
            children = 2 * len(value) if isinstance(value, dict) else len(value) if isinstance(value, list) else 0
            if self.nodes + len(pending) + children > MAX_SCHEMA_NODES:
                raise ValueError("schema node limit exceeded")
            if isinstance(value, dict):
                self.references += "$ref" in value
                if self.references > MAX_SCHEMA_REFERENCES:
                    raise ValueError("schema reference limit exceeded")
                pending.extend((item, depth + 1) for pair in value.items() for item in pair)
            elif isinstance(value, list):
                pending.extend((item, depth + 1) for item in value)
        self.bytes += len(json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))
        if self.bytes > MAX_SCHEMA_BYTES:
            raise ValueError("schema text exceeds the shared 128 KiB limit")
