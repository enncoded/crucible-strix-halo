# origin/main @ c286bda

| Prompt | Spec | Repeat | Prefill tok/s | TTFT s | Decode tok/s | Accept % | Prefill ceiling % | Decode ceiling % | Fused prefill ceiling % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| long_65536 | 0 | 0 | 1185.11 | 55.30 | n/a | n/a | 51.30 | n/a | 30.65 |

TTFT is engine prefill time through the first target ID. It excludes model load and request setup.
Decode excludes the first output ID. No stop token ends the fixed output length.
Unknown ceilings remain n/a. Percentages are analytical throughput ratios, not hardware counter measurements.
