"""Validate static shared-first timing evidence. This is not a shared gate."""
import argparse
import json
import math
import pathlib
import random
import re
import statistics

DECLARATION="SHARED_FIRST C=2 PF=4 cache=default mode0=original mode1=shared R1=unchanged"
EQUALITY=re.compile(r"EQUALITY (R1|R8|B8) mode=1 H=1 logits=1 ids=1")


def number(text):
    value=float(text)
    if not math.isfinite(value) or value<=0 or value>310000:raise ValueError("invalid timing value")
    return value


def put(values,key,value):
    if key in values:raise ValueError(f"duplicate evidence: {key}")
    return {**values,key:value}


def ci(values):
    rng=random.Random(1234)
    samples=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(10000))
    return [samples[249],samples[9749]]


def paired(samples):
    delta=[base-cut for base,cut in samples]
    return {"baseline_ms":statistics.median(b for b,c in samples),
            "candidate_ms":statistics.median(c for b,c in samples),
            "saved_ms":statistics.median(delta),"ci95_ms":ci(delta),
            "samples":[{"baseline_ms":b,"candidate_ms":c,"saved_ms":b-c} for b,c in samples]}


def live(line):
    tokens=line.split();fields=dict(token.split("=",1) for token in tokens[1:])
    if len(fields)!=len(tokens)-1 or fields.get("ids_equal")!="1":raise ValueError("invalid live equality")
    if fields.get("mode") not in {"0","1"}:raise ValueError("invalid live mode")
    if "round" not in fields or not fields["round"].isdigit():raise ValueError("invalid live round")
    kind=tokens[0];round=int(fields["round"])
    if kind=="SHARED_SPEC":
        if round not in range(1,7) or set(fields)!={"mode","round","tok_s","steps","decode_ms","ids_equal"}:raise ValueError("invalid spec fields")
        ms=number(fields["decode_ms"]);tps=number(fields["tok_s"])
        if not fields["steps"].isdigit() or int(fields["steps"])<=0 or abs(tps-111000/ms)>1e-4:raise ValueError("inconsistent spec timing")
    else:
        if round not in {1,2} or set(fields)!={"mode","round","aggregate_tok_s","wall_ms","ids_equal"}:raise ValueError("invalid batch fields")
        ms=number(fields["wall_ms"]);tps=number(fields["aggregate_tok_s"])
        if abs(tps-8000/ms)>1e-4:raise ValueError("inconsistent batch timing")
    return (kind,int(fields["mode"]),round),{"ms":ms,"tps":tps,"fields":fields}


def collect(text):
    steps={};equal={};calls={}
    for line in text.splitlines():
        if line.startswith("EQUALITY "):
            match=EQUALITY.fullmatch(line)
            if not match:raise ValueError("invalid final H/logits/ID equality")
            equal=put(equal,match[1],True)
        elif line.startswith("SHARED_SAMPLES "):
            tokens=line.split();work=tokens[1]
            if work not in {"R8","B8"} or len(tokens)!=34:raise ValueError("expected32 timing pairs")
            samples=[tuple(number(v) for v in token.split("/")) for token in tokens[2:]]
            if any(len(p)!=2 for p in samples):raise ValueError("invalid step pair")
            steps=put(steps,work,paired(samples))
        elif line.startswith("SHARED_SPEC ") or line.startswith("SHARED_BATCH "):
            key,value=live(line);calls=put(calls,key,value)
        elif line.startswith("STOP "):raise ValueError("the candidate failed the R8 kill line")
    return steps,equal,calls


def live_summary(calls,kind,rounds):
    samples=[(calls[kind,0,r]["ms"],calls[kind,1,r]["ms"]) for r in rounds]
    result=paired(samples)
    return {"baseline_tok_s":statistics.median(calls[kind,0,r]["tps"] for r in rounds),
            "candidate_tok_s":statistics.median(calls[kind,1,r]["tps"] for r in rounds),
            "decode_saved_ms":result["saved_ms"],"ci95_ms":result["ci95_ms"],"pairs":result["samples"]}


def parse(text,exit_code):
    if exit_code!=0:raise ValueError("GPU job did not exit zero")
    if DECLARATION not in text:raise ValueError("missing static-order declaration")
    steps,equal,calls=collect(text)
    if set(steps)!={"R8","B8"} or set(equal)!={"R1","R8","B8"}:raise ValueError("incomplete step or equality evidence")
    expected={(kind,m,r) for kind,rounds in [("SHARED_SPEC",range(1,7)),("SHARED_BATCH",[1,2])] for m in [0,1] for r in rounds}
    if set(calls)!=expected:raise ValueError("incomplete live timing evidence")
    if steps["R8"]["saved_ms"]<.5:raise ValueError("R8 saving is below the kill line")
    return {"steps":steps,"equality":equal,"spec":live_summary(calls,"SHARED_SPEC",range(1,7)),
            "batch_live":live_summary(calls,"SHARED_BATCH",[1,2]),
            "bootstrap":{"seed":1234,"repetitions":10000,"interval":"percentile95%"},
            "caveat":"B8 live uses identical prompts. Timing/equality probes are not shared gates."}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("log",type=pathlib.Path)
    parser.add_argument("exit_file",type=pathlib.Path);parser.add_argument("--output",type=pathlib.Path,required=True)
    args=parser.parse_args();result=parse(args.log.read_text(),int(args.exit_file.read_text().strip()))
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":main()
