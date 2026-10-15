# gfx1151 calibration (measured 2026-10-01)

Our own measurements on the Framework Desktop (Ryzen AI Max+ 395, Radeon 8060S,
128 GB LPDDR5X-8000), not spec sheets or third-party articles. Probes are in
`bench/`; the raw outputs are not in this repository. Setup: Arch, kernel 7.2.7, ROCm/HIP
7.2.4 (clang 22), `amdgpu.gttsize=126976`, platform profile `balanced` unless
noted, GPU otherwise idle.

Note: `hipDeviceProp_t::multiProcessorCount` reports **20** on gfx11 (WGPs, not
CUs). Per-CU numbers below use the real 40 CUs.

## Memory

| Measurement | Result |
|---|---|
| Peak cold read, 2 GiB stream (best shape) | **245.5 GB/s** (96% of 256 theoretical) |
| BabelStream HIP (2 GiB arrays, fp64) | Dot 243.7, Copy 237.3, Mul 236.1, Add 233.0, Triad 233.2 GB/s |
| Write-only / copy (read+write counted) | 226.1 / 209.0 GB/s |
| Scattered random chunks over a 48 GiB pool, one kernel | 222-230 GB/s for 64 KiB to 4 MiB chunks |
| MALL re-read, 32 MiB buffer, one kernel incl. launch | 719 GB/s; cliff to DRAM speed at 48 MiB |
| Allocation type (hipMalloc, fine-grained, uncached, host coherent/noncoherent, managed) | identical: L2 153, MALL 642, DRAM 244 GB/s |

"~215 GB/s real" is wrong for this machine: a well-shaped stream sustains 244-245.

**Best read shape:** few waves with deep unrolling. 128 threads x 1 block per WGP
(1 wave per SIMD) with 16 outstanding 16-byte loads per lane: 245.5 GB/s.
Grid-stride beats per-block contiguous chunks (236). Non-temporal loads
(`slc dlc`) neither help nor hurt a cold stream. 168 of 601 shapes exceed 230 GB/s;
the worst (1 wave, no unroll) gets 71.

**One kernel streaming S cold bytes** (event-timed, launch included), i.e. what a
per-matrix kernel gets:

| S | 256 KiB | 1 MiB | 4 MiB | 16 MiB | 64 MiB | 256 MiB | 2 GiB |
|---|---|---|---|---|---|---|---|
| GB/s | 42 | 106-109 | 176-183 | 209-213 | 234 | 240-242 | 244 |
| us | 6.2 | 9.6-9.9 | 23 | 79-80 | 286 | 1,111 | 8,790 |

Fixed cost per kernel is ~5.5 us at small sizes and ~7-12 us at 4-64 MiB (launch,
ramp, tail). Flash-Next's matrices are 0.6-25 MB, which is why per-matrix
kernels land at 40-55% of the bandwidth roofline.

## Latency (one lane, random pointer chase, 128-byte hops)

| Level | Working set | ns/hop |
|---|---|---|
| L0 | 16 KiB | 15 |
| L1 (256 KiB per shader array) | 32-256 KiB | 37-45 |
| L2 (2 MiB) | 512 KiB-2 MiB | 76-116 |
| MALL (32 MiB) | 4-32 MiB | 135-141 |
| DRAM | 64 MiB-2 GiB | 243-252 |
| DRAM + GPU TLB misses | 4 / 8 GiB random | 303 / 522 |

One hop per 4 KiB page: 260 ns at 4 GiB, 447 at 8 GiB, 519 at 16 GiB. TLB reach is
2-4 GiB for random access. For streaming it costs only ~5-8% (scatter test above).

## Synchronization

| Mechanism | Cost |
|---|---|
| Host launch throughput, empty kernels | 1.73 us/launch |
| Device-side gap between dependent kernels (stream) | 1.84-1.88 us |
| Same, HIP graph | 1.72-1.76 us |
| Cross-workgroup one-way signal (atomic flag, agent scope) | **146 ns** |
| Grid barrier, 40 workgroups x 256 threads, relaxed spin + acquire fence | **230 ns** |
| Grid barrier, 80 workgroups x 256 | 314 ns (649 with acquire-load spinning) |

A persistent kernel pays ~0.15-0.3 us per dependency instead of ~1.8 us plus the
ramp/tail of a fresh kernel. Spin with relaxed loads and fence once.

## Compute

WMMA, sustained ~1 s per point, 8 independent accumulator chains per wave:

| Variant | balanced (100 W) | performance (~145 W) |
|---|---|---|
| f16 -> f32 | 48.2 TFLOPS (sclk ~2.55-2.75 GHz) | 54.4 TFLOPS (~2.8-2.9 GHz) |
| bf16 -> f32 | 51.3 | 55.1 |
| f16 -> f16 | 50.5 | |
| iu8 -> i32 | 49.4 TOPS | |
| iu4 -> i32 | 95.5 TOPS | |

- INT8 WMMA = FP16 rate; INT4 = 2x. No FP8 on gfx11.
- At sustained load the `balanced` profile holds package power at 100 W and sclk
  falls to ~2.5-2.7 GHz. `performance` raises it to ~143-154 W for +12% WMMA,
  edge temperature 75-77 C.
- ~478 of the theoretical 512 FLOP/clk/CU are reached.

VALU, instructions per clock per SIMD (1.0 = full rate; sclk from a lagging
hwmon sampler, so read +-10%):

| Rate | Instructions |
|---|---|
| full | v_and/xor, v_mul_u32_u24, v_mad_u32_u24, v_lshl_add_u32, **v_sad_u8**, v_alignbit_b32, v_bfe_u32, v_perm_b32, v_fmac_f32, v_pk_fma_f16, **v_dot2_f32_f16**, v_pk_add_f16, **v_permlanex16_b32**, v_cvt_f32_f16 |
| half | **v_dot4_u32_u8**, v_mov_b32 with DPP row_shr |
| quarter | **v_mul_lo_u32** |

Consequences for EXL3 `mul1` decode: use `v_sad_u8` (full rate) for the byte sum,
not `v_dot4_u32_u8` (half); the state is 16 bits, so the 32-bit constant multiply
splits into full-rate 24-bit multiplies (3 slots instead of 4).

## Probes

`bench/build.sh` builds everything with `hipcc -O3 --offload-arch=gfx1151` and
writes device ISA to `build/isa/`. Run each probe under `timeout`: a wrong peer
index in an early `sync` build spun forever.

- `membw occ|hot|cold|alloc|rw|scatter`
- `latency [stride] [max_bytes]`
- `launch`, `sync`
- `compute valu|wmma`
