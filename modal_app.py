"""Single entrypoint: every stage runs on Modal, all artifacts land in the `marathi-asr` volume.

    modal run modal_app.py --stage prepare     # data + profile      (CPU)
    modal run modal_app.py --stage smoke       # end-to-end smoke    (1 GPU)
    modal run modal_app.py --stage ablations   # base eval + all fine-tune runs in parallel (1 GPU each)
    modal run modal_app.py --stage analyze     # tables, subgroups, error sheet (CPU)
    modal volume get marathi-asr results ./    # pull results locally
"""
import modal

app = modal.App("marathi-bodhan-asr")
vol = modal.Volume.from_name("marathi-asr", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]
V = "/vol"

base = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "libsndfile1", "git", "build-essential")
cpu_image = (base.pip_install("huggingface_hub", "pyarrow", "pandas", "soundfile", "librosa", "matplotlib", "jiwer", "pyyaml", "tabulate")
             .env({"HF_HOME": f"{V}/hf"}).add_local_dir("src", "/root/src").add_local_dir("configs", "/root/configs"))
gpu_image = (base.pip_install("torch==2.7.1", "torchaudio==2.7.1")
             .pip_install("nemo_toolkit[asr]==2.7.3", "huggingface_hub", "pandas", "jiwer", "matplotlib", "pyyaml")
             .env({"HF_HOME": f"{V}/hf"}).add_local_dir("src", "/root/src").add_local_dir("configs", "/root/configs"))
GPU = dict(image=gpu_image, gpu="H100", cpu=16, memory=65536, volumes={V: vol}, secrets=secrets, timeout=3600)  # 1 h cap: a hang must not drain credits


def cfg():
    import yaml
    return yaml.safe_load(open("/root/configs/train.yaml"))


@app.function(image=cpu_image, volumes={V: vol}, secrets=secrets, cpu=8, memory=32768, timeout=3600)
def prepare():
    import os
    from src import data, profile_data
    c = cfg()["data"]
    print(data.build(f"{V}/data", c["train_shards"], c["n_train"], c["dev_frac"], cfg()["seed"]))
    os.makedirs(f"{V}/results", exist_ok=True)
    print(profile_data.run(f"{V}/data", f"{V}/results"))
    vol.commit()


@app.function(image=cpu_image, volumes={V: vol}, secrets=secrets, cpu=4, memory=16384, timeout=3600)
def prepare_shard(shard: int, exclude: set):
    from src import data
    out = data.process_shard(shard, exclude, f"{V}/data")
    vol.commit()
    return out


@app.function(image=cpu_image, volumes={V: vol}, secrets=secrets, cpu=4, memory=16384, timeout=3600)
def prepare_large():
    """Scale-up data: every shard in its own container, in parallel."""
    import json
    import pandas as pd
    exclude = {json.loads(l)["speaker_id"] for s in ["eval", "dev"] for l in open(f"{V}/data/{s}.json")}
    shards = cfg()["data"]["large_shards"]
    lines, rows = [], []
    for ls, rs in prepare_shard.starmap([(s, exclude) for s in shards]):
        lines += ls
        rows += rs
    with open(f"{V}/data/train_large.json", "w") as f:
        f.writelines(json.dumps(l, ensure_ascii=False) + "\n" for l in lines)
    df = pd.DataFrame(rows)
    df["reason"] = df["reason"].fillna("kept")
    df.to_parquet(f"{V}/data/rows_large.parquet")
    m = pd.DataFrame(lines)
    stats = {"shards": len(shards), "raw_examples": len(df), "valid_examples": len(m),
             "removed_by_reason": df.reason[df.reason != "kept"].value_counts().to_dict(),
             "hours": round(m.duration.sum() / 3600, 2), "speakers": m.speaker_id.nunique(),
             "duration_median": round(m.duration.median(), 2), "scenario": m.scenario.value_counts().to_dict()}
    json.dump(stats, open(f"{V}/results/profile_large.json", "w"), indent=2, ensure_ascii=False)
    print(stats)
    vol.commit()


@app.function(image=cpu_image, volumes={V: vol}, secrets=secrets, timeout=1800)
def fetch_model():
    from huggingface_hub import snapshot_download
    snapshot_download("bodhan-ai/indic-transcribe-core", allow_patterns=["nemo/*"])  # into the volume's HF cache
    vol.commit()


@app.function(image=cpu_image, volumes={V: vol}, secrets=secrets, timeout=4 * 3600)
def pipeline(only: str = ""):
    """Everything after the smoke test in one detached call: model download || data prep -> scale-up prep -> runs."""
    model_dl = fetch_model.spawn()
    prepare.remote()
    prepare_large.remote()
    model_dl.get()
    ablations.remote(only or ",".join(cfg()["ablations"]))


@app.function(**GPU)
def smoke():
    from src import smoke_test
    smoke_test.run(cfg(), f"{V}/data", f"{V}/runs/smoke")
    vol.commit()


@app.function(**GPU)
def evaluate_base():
    from src import evaluate, model
    evaluate.run(model.load(), f"{V}/data/eval.json", f"{V}/results/base", cfg())
    vol.commit()


@app.function(**GPU)
def finetune(name: str):
    from src import evaluate, model, train
    ckpt = train.run(cfg(), cfg()["ablations"][name], f"{V}/data", f"{V}/runs/{name}")
    vol.commit()
    # evaluate from the *reloaded* checkpoint, not the in-memory model: proves the artifact is usable
    evaluate.run(model.load(ckpt), f"{V}/data/eval.json", f"{V}/results/{name}", cfg())
    vol.commit()


@app.function(**GPU)
def evaluate_run(name: str):
    """Eval-only for an existing checkpoint (runs/<name>/model.nemo), or the base model for name='base'."""
    from src import evaluate, model
    vol.reload()
    evaluate.run(model.load(None if name == "base" else f"{V}/runs/{name}/model.nemo"), f"{V}/data/eval.json", f"{V}/results/{name}", cfg())
    vol.commit()


@app.function(image=cpu_image, volumes={V: vol}, timeout=4 * 3600)
def ablations(only: str = ""):
    """Remote fan-out, so the parallel runs survive `modal run --detach` and a closed laptop."""
    names = only.split(",") if only else list(cfg()["ablations"])
    calls = ([] if only else [evaluate_base.spawn()]) + [finetune.spawn(n) for n in names]  # one H100 each
    for c in calls:
        c.get()


@app.function(image=cpu_image, volumes={V: vol}, timeout=1800)
def analyze():
    from src import analyze
    analyze.run(f"{V}/results", list(cfg()["ablations"]))
    vol.commit()


@app.local_entrypoint()
def main(stage: str = "prepare", only: str = ""):
    if stage == "prepare":
        prepare.remote()
    elif stage == "pipeline":
        pipeline.remote(only)
    elif stage == "prepare_large":
        prepare_large.remote()
    elif stage == "smoke":
        smoke.remote()
    elif stage == "ablations":
        ablations.remote(only)
    elif stage == "eval":  # --only base,A3_half_data
        list(evaluate_run.map(only.split(",")))
    elif stage == "analyze":
        analyze.remote()
