# Draft test data

- The `.ids` files contain the seven chat prompts from `tools/draft_prepare.py`.
- `blocks*.txt` contains aligned EXL3 block IDs from the source-code corpus.
- `tokens*.txt` contains individual token IDs from the source-code corpus.
- `mixed-tokens*.txt` also uses the Databricks Dolly 15K corpus.
- The manifest files contain file counts, token counts, and corpus coverage.
- No generated continuation is part of either corpus.

## Mixed-corpus attribution

Copyright (2023) Databricks, Inc.

The mixed subsets use frequency counts from [Databricks Dolly 15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k).

The dataset uses the [CC BY-SA 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/legalcode).

Some dataset text comes from Wikipedia. Copyright belongs to the Wikipedia editors and contributors.

The raw dataset is not included here. The `mixed-tokens*.txt` files contain only selected tokenizer IDs.

The downloaded JSONL file has this SHA-256 value:

```text
2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec
```

Use this command after downloading the dataset:

```sh
python tools/draft_prepare.py CKPT OUTPUT --chat-jsonl dolly.jsonl
```

The corpus contains local source files. Changes to those files or the Python version can change the selected IDs.

Use the committed selection files to reproduce the reported head contents exactly.
