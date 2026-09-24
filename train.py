"""Fine-tune with NeMo's own Lightning module + lhotse dataloader (same path as speech_to_text_finetune.py,
but in-process so ablations are just dict overrides)."""
import json
import os
import re
import time

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

from src import model


def ds_cfg(manifest, t, shuffle, seed):
    return OmegaConf.create({
        "use_lhotse": True, "manifest_filepath": manifest, "text_field": "answer", "lang_field": "target_lang",
        "batch_duration": t["batch_duration"], "use_bucketing": False, "shuffle": shuffle, "seed": seed,
        "shuffle_buffer_size": 10000, "num_workers": 12, "pin_memory": True, "min_duration": 0.3, "max_duration": 30.0,
    })


def unfreeze_top(m, k):
    """Freeze everything except the last k encoder layers, the last k decoder layers and the output head."""
    layer = lambda prefix: {int(x) for n, _ in m.named_parameters() for x in re.findall(rf"^{prefix}\.(\d+)\.", n)}
    enc, dec = layer(r"encoder\.layers"), layer(r"transf_decoder\._decoder\.layers")
    top = [rf"^encoder\.layers\.({'|'.join(map(str, sorted(enc)[-k:]))})\.",
           rf"^transf_decoder\._decoder\.layers\.({'|'.join(map(str, sorted(dec)[-k:]))})\.",
           r"^transf_decoder\._decoder\.final_layer_norm", r"^log_softmax\."]
    for n, p in m.named_parameters():
        p.requires_grad_(any(re.search(pat, n) for pat in top))
    assert enc and dec, "layer naming changed; nothing would be unfrozen"
    print(f"unfrozen: encoder layers {sorted(enc)[-k:]}, decoder layers {sorted(dec)[-k:]}, head")


def run(c, overrides, data_dir, out_dir):
    t = {**c["training"], **overrides}
    os.makedirs(out_dir, exist_ok=True)
    pl.seed_everything(c["seed"])
    m = model.load().train()  # load() returns eval mode; Lightning>=2.2 keeps it -> SpecAugment/dropout would be OFF

    if t.get("freeze_encoder"):
        m.encoder.requires_grad_(False)  # SpecAugment (preprocessor side) still applies; encoder dropout stays on
    if t.get("unfreeze_last"):
        unfreeze_top(m, t["unfreeze_last"])
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)

    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision=t["precision"], max_steps=t["max_steps"],
        max_time={"minutes": t["max_minutes"]} if t.get("max_minutes") else None,  # hard wall-clock cap
        val_check_interval=t["val_every_n_steps"], check_val_every_n_epoch=None, num_sanity_val_steps=0,
        gradient_clip_val=t["grad_clip"], use_distributed_sampler=False,  # lhotse does its own sampling
        logger=CSVLogger(out_dir, name="", version=""), callbacks=[LearningRateMonitor("step")],
        enable_checkpointing=False, log_every_n_steps=10,
    )
    m.set_trainer(trainer)
    m.setup_training_data(ds_cfg(f"{data_dir}/{t.get('train_manifest', 'train')}.json", t, True, c["seed"]))
    m.setup_validation_data(ds_cfg(f"{data_dir}/{t.get('dev_manifest', 'dev')}.json", t, False, c["seed"]))
    m.setup_optimization(OmegaConf.create({
        "name": "adamw", "lr": t["lr"], "weight_decay": t["weight_decay"], "betas": [0.9, 0.98],
        "sched": {"name": "CosineAnnealing", "warmup_steps": t["warmup_steps"], "max_steps": t["max_steps"], "min_lr": t["lr"] / 20},
    }))

    start = time.time()
    trainer.validate(m)  # step-0 dev loss = the base model's reference point on the curve
    trainer.fit(m)
    ckpt = f"{out_dir}/model.nemo"
    m.save_to(ckpt)
    info = {"config": t, "trainable_params": trainable, "total_params": sum(p.numel() for p in m.parameters()),
            "runtime_min": round((time.time() - start) / 60, 1),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1), "gpu": torch.cuda.get_device_name()}
    json.dump(info, open(f"{out_dir}/run_info.json", "w"), indent=2)
    print(info)
    return ckpt
