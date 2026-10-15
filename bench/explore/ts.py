#!/usr/bin/env python3
"""Prefixes each stdin line with CLOCK_MONOTONIC seconds (same clock as kfd_trace)."""
import sys
import time

for line in sys.stdin:
    sys.stdout.write(f"{time.monotonic():.6f} {line}")
    sys.stdout.flush()
