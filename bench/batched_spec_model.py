"""CPU planning model. This does not predict a measured engine rate."""
from dataclasses import dataclass
from math import isfinite, log, log1p

DRAM_BW = 245.5e9
UNION_B8 = 63.575521
EXPERT_BYTES = 1_862_412
TRUNK_BYTES = 3_721_379_812
GDN_STATE_BYTES = 226_492_416
MTP_BASE_BYTES = 83_675_688 + 397_813_764 * 32768 / 248320
MTP_STATE_BYTES = 4_196_352
EXPONENT = log(log1p(-UNION_B8 / 512) / log1p(-10 / 512)) / log(8)

@dataclass(frozen=True)
class Estimate:
    projected_rows: int
    model_union: float
    floor_ms: float
    aggregate_ceiling: float
    expected_60: float
    per_stream_60: float

def union(streams: int, temporal_union: float) -> float:
    return 512 * (1 - (1 - temporal_union / 512) ** (streams ** EXPONENT))

def plan_rate(streams: int, rows: int, acceptance: float, temporal_union: float,
              *, draft_shared: bool = True) -> Estimate:
    if (type(streams) is not int or type(rows) is not int or not 1 <= streams <= 8 or
            not 1 <= rows <= 8 or streams * rows > 16):
        raise ValueError('require B in 1..8, R in 1..8, and B*R <= 16')
    if (type(draft_shared) is not bool or type(acceptance) not in (int, float) or
            type(temporal_union) not in (int, float) or not isfinite(acceptance) or not 0 <= acceptance <= 1 or
            not isfinite(temporal_union) or not 10 <= temporal_union <= min(512, 10 * rows)):
        raise ValueError('invalid acceptance or temporal union')
    model_union = union(streams, temporal_union)
    verify = TRUNK_BYTES + 48 * EXPERT_BYTES * model_union + streams * GDN_STATE_BYTES
    draft = (MTP_BASE_BYTES + EXPERT_BYTES * union(streams, 10) + streams * MTP_STATE_BYTES
             if draft_shared else streams * (MTP_BASE_BYTES + EXPERT_BYTES * 10 + MTP_STATE_BYTES))
    floor = (verify + (rows - 1) * draft) / DRAM_BW
    emitted = streams * sum(acceptance ** depth for depth in range(rows))
    ceiling = emitted / floor
    return Estimate(streams * rows, model_union, floor * 1000, ceiling, ceiling * 0.6, ceiling * 0.6 / streams)
