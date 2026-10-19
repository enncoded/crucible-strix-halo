"""TTFT-first choice between one prompt chunk and one packed decode step.

Pauses between chunks never change bits: each request keeps its own fixed chunk schedule.
"""
from dataclasses import dataclass

DECODE_MAX_SECONDS = 0.25
AGING_FACTOR = 2.0
AGING_FLOOR_SECONDS = 1.0
# Load-adaptive decode (G6c and G6d). A lone decoder runs solo spec steps. From two
# decoders, one packed verification pass serves all of them and verifies each one's drafts within an 8-row
# budget: at k = 2 it gives 1.34 to 1.37 times the per-session speed of solo steps taken in turn.
# With the G6c packed step (`--packed-op step`), enter at 3 instead: there k = 2 packing only breaks even.
PACK_ENTER = 2
PACK_EXIT = 1
# Decode passes between prefill layers (engine/runtime/prefill_interleave.h): the engine's layer count and the most tokens
# plus done event that one pass can publish for a request.
MODEL_LAYERS = 48
PASS_EVENTS = 9
# A chunk with fewer rows runs without passes: its 5 passes cost the turn
# about 0.2 s of TTFT (live t15 data). A peer then waits up to decode_max (0.25 s) of earlier commands, the chunk
# (about 0.11 s + 1.03 ms per row, a little more per row at 200K+) and its own step: about 0.7 s at 299 rows, near
# the 0.7 s of 4,096-row chunks with passes today (at 511 rows it would be about 0.9 s). Passes never change
# bits; 0 restores passes in every bulk chunk. A rule on time and depth is a train-16 follow-up.
INTERLEAVE_MIN_ROWS = 300
# One encoder slice (engine/model/flashnext/vision.h kSliceWork, about 60-80 ms): 2.196 s for the 29 slices of a 2560x1440
# image on gfx1151 (night 1). The worker updates it from the slice commands it runs.
ENCODER_SLICE_SECONDS = 0.076
# Each measured slice is held to this band before it enters the average: one long command (a stall, a slow
# file read) moves the average a little, never by 10 times.
ENCODER_SLICE_BAND = (0.02, 0.2)
# The most slices that count: 256 images of 29 slices (2560x1440) are 7,424.
ENCODER_SLICES_MAX = 1 << 13


def encoder_rows(slices, slice_seconds, rate):
    """Prompt rows that cost the device time of the encoder slices that a turn still owes.

    SRPT and aging then rank an image turn by all its work: a 2560x1440 image adds about 3,300 rows at 1,500 rows/s, so
    a text turn of 5,000 rows goes before a 3,600-row image turn (3.3 s against 4.5 s). Order never changes bits.
    """
    return round(min(slices, ENCODER_SLICES_MAX) * slice_seconds * rate) if slices > 0 else 0


def interleave_passes(every, layers=MODEL_LAYERS):
    """Passes inside one bulk chunk; mirrors InterleaveAfterLayer: after layer 1 at the earliest, never after the last,
    half an interval off the chunk edges."""
    return sum(1 for layer in range(1, layers - 1) if (layer + 1 + every // 2) % every == 0) if every > 0 else 0


@dataclass(frozen=True)
class PendingPrefill:
    request_id: int
    remaining: int
    admitted_at: float
    # A think-cap continuation: its client already has the turn's first token (server/http.py close_thinking).
    continuation: bool = False


def choose(now, ready, pending, last_step, rate, *, decode_max=DECODE_MAX_SECONDS,
           aging=AGING_FACTOR, aging_floor=AGING_FLOOR_SECONDS):
    """Return ("step", ids), ("prefill", id), or None.

    Waiting turns go first because TTFT is the primary metric. Decoders still get one step
    whenever decode_max has passed since their last step, so a long prefill cannot starve them.
    A chunk is never split, so the real bound is max(decode_max, one chunk): about 3 s at 4096 rows.
    """
    if not pending:
        return ("step", list(ready)) if ready else None
    if ready and now - last_step >= decode_max:
        return ("step", list(ready))
    return ("prefill", _next_prefill(now, pending, rate, aging, aging_floor))


def packed_mode(current, decoders, *, enter=PACK_ENTER, leave=PACK_EXIT):
    """Return True when every decoder, speculative or not, should share one packed pass.

    Output does not depend on the mode (greedy, and the keyed sampling law), so only speed changes.
    """
    if decoders >= enter: return True
    if decoders <= leave: return False
    return current


def _next_prefill(now, pending, rate, aging, aging_floor):
    # A continuation is the middle of a turn that already streams, and it is short (it resumes its first job's
    # checkpoint), so it goes before every new turn, aged or not. Without this a peer's aged prefill held the
    # stream for up to 30 s after its first token (decode gaps).
    continuations = [item for item in pending if item.continuation]
    if continuations:
        return min(continuations, key=lambda item: (item.admitted_at, item.request_id)).request_id
    # Shortest remaining first minimizes mean TTFT. A turn that has waited longer than
    # aging times its own estimated prefill moves ahead, so a long tool output still finishes.
    def deadline(item):
        return item.admitted_at + aging_floor + aging * item.remaining / rate
    aged = [item for item in pending if now >= deadline(item)]
    if aged:
        return min(aged, key=lambda item: (item.admitted_at, item.request_id)).request_id
    return min(pending, key=lambda item: (item.remaining, item.admitted_at, item.request_id)).request_id
