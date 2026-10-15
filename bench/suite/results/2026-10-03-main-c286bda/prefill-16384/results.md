# origin/main @ c286bda

| Prompt | Spec | Repeat | Prefill tok/s | TTFT s | Decode tok/s | Accept % | Prefill ceiling % | Decode ceiling % | Fused prefill ceiling % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| long_16384 | 0 | 0 | 1275.75 | 12.84 | n/a | n/a | 55.23 | n/a | 33.00 |

TTFT is engine prefill time through the first target ID. It excludes model load and request setup.
Decode excludes the first output ID. No stop token ends the fixed output length.
Unknown ceilings remain n/a. Percentages are analytical throughput ratios, not hardware counter measurements.
