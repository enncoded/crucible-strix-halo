#!/usr/bin/env python3
"""Record read-only memory and GPU telemetry once per second."""
import glob
import json
import pathlib
import subprocess
import sys
import time


def text(path):
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError as error:
        return f"ERROR: {error}"


def counters(path):
    result = {}
    for line in text(path).splitlines():
        parts = line.replace(":", "").split()
        if len(parts) > 1 and parts[1].isdigit():
            result[parts[0]] = int(parts[1])
    return result


def large_processes():
    result = []
    for path in glob.glob("/proc/[0-9]*/status"):
        status = text(path)
        fields = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                  for line in status.splitlines() if ":" in line}
        rss = int(fields.get("VmRSS", "0").split()[0])
        if rss >= 1 << 20:
            result.append({"pid": fields.get("Pid"), "name": fields.get("Name"), "rss_kib": rss})
    return result


def kernel_workers():
    result = {}
    for path in glob.glob("/proc/[0-9]*/comm"):
        name = text(path)
        if name.startswith(("kcompactd", "khugepaged", "kswapd")):
            fields = text(str(pathlib.Path(path).with_name("stat"))).rsplit(")", 1)[-1].split()
            if len(fields) > 12:
                result[name] = int(fields[11]) + int(fields[12])
    return result


def kfd_counters():
    paths = glob.glob("/sys/class/kfd/kfd/proc/*/stats_*/evicted_ms")
    paths += glob.glob("/sys/class/kfd/kfd/proc/*/counters_*/*")
    return {p: text(p) for p in paths}


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: qsa_memwatch.py OUTPUT_JSONL")
    device = "/sys/class/drm/card1/device"
    paths = [f"{device}/mem_info_gtt_used", f"{device}/mem_info_vram_used", f"{device}/pp_dpm_sclk"]
    paths += glob.glob(f"{device}/hwmon/hwmon*/power1_average")
    paths += glob.glob(f"{device}/hwmon/hwmon*/power1_input")
    paths += glob.glob(f"{device}/hwmon/hwmon*/freq1_input")
    with open(sys.argv[1], "w", buffering=1) as output:
        while True:
            sample = {"epoch_s": time.time(), "monotonic_s": time.monotonic(),
                      "meminfo_kib": counters("/proc/meminfo"), "vmstat": counters("/proc/vmstat"),
                      "memory_pressure": text("/proc/pressure/memory"), "gpu": {p: text(p) for p in paths},
                      "large_processes": large_processes(), "kernel_worker_ticks": kernel_workers(),
                      "kfd_counters": kfd_counters(),
                      "free_g": subprocess.run(["free", "-g"], check=True, capture_output=True, text=True).stdout}
            output.write(json.dumps(sample) + "\n")
            time.sleep(1)


if __name__ == "__main__":
    main()
