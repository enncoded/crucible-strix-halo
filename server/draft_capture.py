"""Draft vocabulary data from served turns. Off unless the server gets --draft-capture DIR.

counts: per day and turn class, how often each 128-token block occurs in the generated output. The file holds
  numbers only (draft-counts-YYYY-MM-DD.json). Prompts are not counted.
tokens: per day, the prompt and output token IDs of each finished turn in a binary spool
  (draft-tokens-YYYY-MM-DD.bin) for the machine-only training pipeline. It needs the user's explicit OK.

Both modes write no text, no request ID and no time finer than the UTC day (file times are set to the day's
00:00 UTC). The directory must be the server user's own and private (0700, no symlink); every file is 0600. One
server at a time writes a directory (DIR/.lock). The spool stops at max_bytes over all its files; later turns are
dropped and counted.

Export floor: the per-day files never leave the serving machine. Any export sums 7 or more days and drops each class with
fewer than 20 sequences.
"""
import calendar
import fcntl
import json
import os
import stat
import struct
import tempfile
import threading
import time
from pathlib import Path

VOCAB, BLOCK = 248320, 128
BLOCKS = VOCAB // BLOCK
MODES = ("counts", "tokens")
WRITE_SECONDS = 600
SPOOL_MAGIC = b"DCT1"
RECORD = struct.Struct("<BII")             # class index, prompt length, output length; then uint32 IDs
CLASSES = ("chat", "chat+think", "tool", "tool+think")


def turn_class(tools, thinking):
    return ("tool" if tools else "chat") + ("+think" if thinking else "")


def _check_ids(ids):
    for token in ids:
        if type(token) is not int or not 0 <= token < VOCAB:
            raise ValueError("draft capture needs token IDs in [0, 248320)")


def _open_private(path, flags):
    fd = os.open(path, flags | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)
    return os.fdopen(fd, "wb")


def _private_dir(directory):
    """Make or check the capture directory: a real directory of this user, closed to group and other."""
    if not directory.exists() and not directory.is_symlink():
        directory.mkdir(mode=0o700, parents=True)
    info = os.lstat(directory)
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("the draft capture directory must be a directory, not a symlink")
    if info.st_uid != os.geteuid():
        raise ValueError("the draft capture directory must belong to the server user")
    if info.st_mode & 0o077:
        raise ValueError("the draft capture directory must be closed to group and other (0700)")


def _valid_counts(doc, day):
    """True when doc is a counts document of this day with the expected shape (numbers only)."""
    def count(v):
        return type(v) is int and v >= 0
    if not isinstance(doc, dict) or set(doc) != {"day", "granularity", "vocab", "block", "categories"}:
        return False
    if (doc["day"], doc["granularity"], doc["vocab"], doc["block"]) != (day, "block", VOCAB, BLOCK):
        return False
    cats = doc["categories"]
    return isinstance(cats, dict) and all(
        name in CLASSES and isinstance(entry, dict) and set(entry) == {"sequences", "tokens", "counts"}
        and count(entry["sequences"]) and count(entry["tokens"]) and isinstance(entry["counts"], list)
        and len(entry["counts"]) == BLOCKS and all(count(c) for c in entry["counts"])
        for name, entry in cats.items())


class DraftCapture:
    def __init__(self, directory, *, mode, max_bytes=8 << 30, clock=time.time):
        # mode has no default: only the exact value "tokens" writes token IDs (the user's approval).
        directory = Path(directory)
        if not directory.is_absolute():
            raise ValueError("the draft capture directory must be an absolute path")
        if mode not in MODES:
            raise ValueError("the draft capture mode is counts or tokens")
        self.dir, self.mode, self.max_bytes, self.clock = directory, mode, int(max_bytes), clock
        _private_dir(self.dir)
        self.lock_fd = os.open(self.dir / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.lock_fd)
            raise RuntimeError("another server writes this draft capture directory") from None
        self.lock = threading.Lock()
        self.day, self.doc, self.last_write, self.dropped = None, None, 0.0, 0

    def _today(self):
        return time.strftime("%Y-%m-%d", time.gmtime(self.clock()))

    def record(self, *, tools, thinking, prompt_ids, output_ids):
        output_ids = list(output_ids)
        _check_ids(output_ids)
        if self.mode == "tokens":       # counts mode never reads or copies the prompt
            prompt_ids = list(prompt_ids)
            _check_ids(prompt_ids)
        with self.lock:
            day = self._today()
            if self.mode == "tokens":
                self._spool(day, CLASSES.index(turn_class(tools, thinking)), prompt_ids, output_ids)
                return
            if day != self.day:
                self._flush()
                self.day, self.doc = day, self._load(day)
            entry = self.doc["categories"].setdefault(
                turn_class(tools, thinking), {"sequences": 0, "tokens": 0, "counts": [0] * BLOCKS})
            entry["sequences"] += 1
            entry["tokens"] += len(output_ids)
            for token in output_ids:
                entry["counts"][token // BLOCK] += 1
            if self.clock() - self.last_write >= WRITE_SECONDS:
                self._flush()

    def _counts_path(self, day):
        return self.dir / f"draft-counts-{day}.json"

    def _day_start(self, day):
        return calendar.timegm(time.strptime(day, "%Y-%m-%d"))

    def _load(self, day):
        """The day's sums so far (a restart adds to them), or an empty document. A file of another shape moves
        to NAME.bad and the day starts empty."""
        path = self._counts_path(day)
        if path.exists():
            try:
                doc = json.loads(path.read_text())
            except (OSError, ValueError):
                doc = None
            if _valid_counts(doc, day):
                return doc
            os.replace(path, path.with_name(path.name + ".bad"))
        return {"day": day, "granularity": "block", "vocab": VOCAB, "block": BLOCK, "categories": {}}

    def _flush(self):
        if self.doc is None:
            return
        path = self._counts_path(self.day)
        fd, temp = tempfile.mkstemp(dir=self.dir, prefix=".draft-counts-", suffix=".tmp")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(json.dumps(self.doc, separators=(",", ":")).encode())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        except BaseException:
            if os.path.exists(temp):
                os.unlink(temp)
            raise
        start = self._day_start(self.day)
        os.utime(path, (start, start), follow_symlinks=False)
        self.last_write = self.clock()

    def _spool_bytes(self):
        return sum(p.stat().st_size for p in self.dir.glob("draft-tokens-*.bin"))

    def _spool(self, day, cls, prompt_ids, output_ids):
        if self.mode != "tokens":
            raise RuntimeError("only the tokens mode writes token IDs")
        path = self.dir / f"draft-tokens-{day}.bin"
        new = not path.exists()
        size = (len(SPOOL_MAGIC) if new else 0) + RECORD.size + 4 * (len(prompt_ids) + len(output_ids))
        if self._spool_bytes() + size > self.max_bytes:
            self.dropped += 1
            return
        data = (SPOOL_MAGIC if new else b"") + RECORD.pack(cls, len(prompt_ids), len(output_ids))
        data += struct.pack(f"<{len(prompt_ids) + len(output_ids)}I", *prompt_ids, *output_ids)
        with _open_private(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        start = self._day_start(day)
        os.utime(path, (start, start), follow_symlinks=False)

    def close(self):
        with self.lock:
            try:
                self._flush()
            finally:
                if self.lock_fd is not None:
                    os.close(self.lock_fd)       # closing the descriptor releases the flock
                    self.lock_fd = None


def read_spool(path):
    """Yield (class, prompt IDs, output IDs) of a spool file. For the test machine pipeline only: never print them."""
    with open(path, "rb") as handle:
        if handle.read(len(SPOOL_MAGIC)) != SPOOL_MAGIC:
            raise ValueError("not a draft token spool")
        while header := handle.read(RECORD.size):
            if len(header) != RECORD.size:
                raise ValueError("truncated draft token spool")
            cls, n_prompt, n_output = RECORD.unpack(header)
            body = handle.read(4 * (n_prompt + n_output))
            if len(body) != 4 * (n_prompt + n_output) or cls >= len(CLASSES):
                raise ValueError("truncated draft token spool")
            ids = list(struct.unpack(f"<{n_prompt + n_output}I", body))
            yield CLASSES[cls], ids[:n_prompt], ids[n_prompt:]
