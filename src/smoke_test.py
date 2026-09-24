"""Smoke test: 16 examples through the *real* training code for 3 steps, then save -> reload -> infer.
Catches config/dataloader/prompt-format/checkpoint problems before committing GPU-hours."""
import json

import torch

from src import model, train


def run(c, data_dir, out_dir):
    lines = open(f"{data_dir}/train.json").readlines()
    open(f"{data_dir}/smoke_train.json", "w").writelines(lines[:16])
    open(f"{data_dir}/smoke_dev.json", "w").writelines(lines[16:24])

    ckpt = train.run(c, {"train_manifest": "smoke_train", "dev_manifest": "smoke_dev", "max_steps": 3,
                         "val_every_n_steps": 3, "warmup_steps": 1, "batch_duration": 60, "lr": 1e-4}, data_dir, out_dir)

    base, tuned = model.load(), model.load(ckpt)  # reload from disk
    changed = sum(not torch.equal(a, b) for a, b in zip(base.parameters(), tuned.parameters()))
    assert changed, "weights did not change after optimizer steps"
    print(f"{changed} parameter tensors changed after 3 steps")

    open(f"{data_dir}/smoke_infer.json", "w").writelines(lines[:4])
    items = [json.loads(l) for l in lines[:4]]
    for it, hyp in zip(items, model.transcribe(tuned, f"{data_dir}/smoke_infer.json", c)):
        print(f"REF: {it['text']}\nHYP: {hyp}\n")
    print("SMOKE TEST PASSED")
