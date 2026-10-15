"""Explicit throughput models. These estimates are not hardware-counter measurements."""
from pathlib import Path

CHECKPOINT_305 = "Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5"
DRAM_GBS = 245.5
STATIC_GB = 3.726
EXPERT_GB_PER_UNION = 0.0894
FULL_MTP_GB = 0.500118692
# Representative consecutive-token unions measured on 2026-10-03.
UNIONS = {1: 10, 2: 16, 3: 22.4, 4: 26.2, 5: 32, 6: 37, 7: 41, 8: 44.1}
PREFILL_CURRENT = 2310
PREFILL_FUSED = 3866


def spec_floor(row):
    depth, forwards = row["spec"], row["forwards"]
    if not depth or not forwards or depth + 1 not in UNIONS or row["proposed"] != forwards * depth:
        return None
    union = UNIONS[depth + 1]
    verify_gb = STATIC_GB + union * EXPERT_GB_PER_UNION
    step_seconds = (verify_gb + depth * FULL_MTP_GB) / DRAM_GBS
    emitted_per_step = (len(row["output_ids"]) - 1) / forwards
    return {"kind": "representative expert union, full-head MTP weight-read floor",
            "rows": depth + 1, "experts_per_layer": union, "verify_gb": verify_gb,
            "draft_gb_per_token": FULL_MTP_GB, "step_floor_ms": 1000 * step_seconds,
            "ceiling_tps": emitted_per_step / step_seconds,
            "excludes": "KV/state traffic, activation traffic, launch gaps; union is not measured on this prompt"}


def annotate_row(row, ceilings, known_checkpoint):
    result = {**row}
    if known_checkpoint and row["spec"] and ceilings.get("spec_tps") is None:
        model = spec_floor(row)
        if model:
            result = {**result, "spec_weight_model": model, "decode_ceiling_tps": model["ceiling_tps"],
                      "decode_ceiling_pct": 100 * row["decode_tps"] / model["ceiling_tps"]}
    if ceilings.get("prefill_tps") == PREFILL_CURRENT:
        result = {**result, "prefill_ceiling_pct": 100 * row["prefill_tps"] / PREFILL_CURRENT,
                  "prefill_fused_ceiling_pct": 100 * row["prefill_tps"] / PREFILL_FUSED}
    return result


def annotate(report):
    known = Path(report["checkpoint"]).name == CHECKPOINT_305
    ceilings = report["ceilings"]
    if ceilings.get("prefill_tps") == PREFILL_CURRENT:
        ceilings = {**ceilings, "prefill_fused_tps": PREFILL_FUSED}
    return {**report, "ceilings": ceilings,
            "rows": [annotate_row(row, ceilings, known) for row in report["rows"]]}
