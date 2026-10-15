# origin/main @ c286bda

| Prompt | Spec | Repeat | Prefill tok/s | TTFT s | Decode tok/s | Accept % | Prefill ceiling % | Decode ceiling % | Fused prefill ceiling % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| code_fibonacci | 0 | 0 | 353.10 | 0.14 | 34.31 | n/a | 15.29 | 64.50 | 9.13 |
| code_fibonacci | 2 | 0 | 329.19 | 0.15 | 65.12 | 91.96 | 14.25 | 62.86 | 8.51 |
| chat_planning | 0 | 0 | 408.95 | 0.15 | 34.28 | n/a | 17.70 | 64.45 | 10.58 |
| chat_planning | 2 | 0 | 377.13 | 0.17 | 49.35 | 55.26 | 16.33 | 64.66 | 9.76 |
| prose_story | 0 | 0 | 401.08 | 0.15 | 34.31 | n/a | 17.36 | 64.49 | 10.37 |
| prose_story | 2 | 0 | 372.19 | 0.17 | 50.34 | 56.00 | 16.11 | 65.08 | 9.63 |
| copy_python | 0 | 0 | 719.94 | 0.29 | 34.31 | n/a | 31.17 | 64.49 | 18.62 |
| copy_python | 2 | 0 | 639.37 | 0.32 | 67.44 | 99.07 | 27.68 | 62.77 | 16.54 |

TTFT is engine prefill time through the first target ID. It excludes model load and request setup.
Decode excludes the first output ID. No stop token ends the fixed output length.
Unknown ceilings remain n/a. Percentages are analytical throughput ratios, not hardware counter measurements.
