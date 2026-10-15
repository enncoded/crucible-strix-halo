#!/usr/bin/env python3
"""ExLlamaV3 reference for image prompts: tower embeddings and greedy IDs.

Run on the test machine inside the refenv runtime (source ~/crucible-work/refenv/env.sh) and under gpu-run.
Each image is first resized by the Crucible/Halogen rule, so ExLlamaV3's own resize keeps it.

  python tools/vision_exllama_ref.py CKPT CASES.json OUTDIR [--new-tokens 32]

CASES.json: [{"name": ..., "image": PATH, "question": ...}, ...]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import vision_ref  # noqa: E402

GEN_PROMPT = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ckpt")
    p.add_argument("cases")
    p.add_argument("out")
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--cache-tokens", type=int, default=16384)
    args = p.parse_args()
    from PIL import Image
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
    from exllamav3.generator.sampler import ArgmaxSampler

    os.makedirs(args.out, exist_ok=True)
    cases = json.load(open(args.cases))
    config = Config.from_directory(args.ckpt)
    vision = Model.from_config(config, component="vision")
    vision.load()
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=args.cache_tokens)
    model.load()
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
    results = []
    for case in cases:
        rgb = vision_ref.resize_rgb(Image.open(os.path.join(os.path.dirname(args.cases), case["image"])))
        image = Image.fromarray(rgb)
        start = time.perf_counter()
        mme = vision.get_image_embeddings(tokenizer=tokenizer, image=image)
        torch.cuda.synchronize()
        encode_s = time.perf_counter() - start
        np.save(os.path.join(args.out, case["name"] + ".exl3.npy"), mme.embeddings.float().cpu().numpy())
        prompt = "<|im_start|>user\n" + mme.text_alias + case["question"] + GEN_PROMPT
        ids = tokenizer.encode(prompt, encode_special_tokens=True, embeddings=[mme])
        job = Job(input_ids=ids, max_new_tokens=args.new_tokens, sampler=ArgmaxSampler(), embeddings=[mme],
                  stop_conditions=[248046, 248044], return_logits=False)
        generator.enqueue(job)
        tokens = []
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                if "token_ids" in result:
                    tokens.extend(result["token_ids"].flatten().tolist())
        entry = dict(name=case["name"], height=rgb.shape[0], width=rgb.shape[1],
                     image_tokens=int(mme.embeddings.shape[0]), prompt_tokens=int(ids.shape[-1]),
                     encode_seconds=encode_s, ids=tokens, text=tokenizer.decode(torch.tensor(tokens)))
        results.append(entry)
        print(json.dumps(entry), flush=True)
    json.dump(results, open(os.path.join(args.out, "exllama.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
