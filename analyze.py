"""Cross-run analysis: ablation table, subgroup deltas, training curves, and an error sheet for manual labelling."""
import glob
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import jiwer
import matplotlib.pyplot as plt
import pandas as pd

from src.text import DEVANAGARI_ONLY, normalize


def load(results, name):
    return json.load(open(f"{results}/{name}/metrics.json")), pd.read_json(f"{results}/{name}/predictions.json")


def ablation_table(results, runs, runs_dir):
    rows = []
    for name in ["base"] + runs:
        if not os.path.exists(f"{results}/{name}/metrics.json"):
            continue
        m, p = load(results, name)
        nospace = lambda xs: [normalize(x).replace(" ", "") or "∅" for x in xs]  # segmentation-invariant CER
        info = json.load(open(f"{runs_dir}/{name}/run_info.json")) if name != "base" else {}
        sub = m["subgroups"]["register"]
        rows.append({"run": name, "WER raw": m["raw"]["wer"], "CER raw": m["raw"]["cer"],
                     "WER norm": m["normalized"]["wer"], "CER norm": m["normalized"]["cer"],
                     "CER no-space": round(100 * jiwer.cer(nospace(p.text), nospace(p.hyp.fillna(""))), 2),
                     "WER read": sub.get("read", {}).get("wer"), "WER spont.": sub.get("spontaneous", {}).get("wer"),
                     "trainable M": round(info.get("trainable_params", 0) / 1e6) or "-",
                     "train min": info.get("runtime_min", "-"), "peak GB": info.get("peak_gpu_mem_gb", "-")})
    return pd.DataFrame(rows)


def subgroup_table(results, run):
    b, f = load(results, "base")[0]["subgroups"], load(results, run)[0]["subgroups"]
    return pd.DataFrame([{"axis": ax, "group": g, "n": b[ax][g]["n"], "WER base": b[ax][g]["wer"], f"WER {run}": f[ax][g]["wer"],
                          "ΔWER": round(f[ax][g]["wer"] - b[ax][g]["wer"], 2)} for ax in b for g in b[ax]])


def tags(ref, ft):
    """Cheap, data-driven hints to speed up manual labelling; the final category is assigned by hand."""
    r, h = normalize(ref).split(), normalize(ft).split()
    t = []
    if len(r) <= 2: t.append("very short ref")
    if len(r) >= 25: t.append("long utterance")
    if len(h) < 0.6 * len(r): t.append("deletions")
    if len(h) > 1.4 * len(r) + 1: t.append("insertions/hallucination")
    if ft and not DEVANAGARI_ONLY.match(normalize(ft) or "क"): t.append("non-Devanagari output")
    if sorted(r) == sorted(h) and r != h: t.append("word order")
    return ", ".join(t)


def error_sheet(results, run, n=50, seed=42):
    _, b = load(results, "base")
    _, f = load(results, run)
    df = b[["audio_filepath", "text", "duration", "scenario", "hyp"]].rename(columns={"hyp": "base"})
    df["finetuned"] = f.set_index("audio_filepath").loc[df.audio_filepath, "hyp"].values
    w = lambda h: [jiwer.wer(normalize(r) or "∅", normalize(x)) for r, x in zip(df.text, df[h])]
    df["wer_base"], df["wer_ft"] = w("base"), w("finetuned")
    df["delta"] = df.wer_ft - df.wer_base
    err = df[df.wer_ft > 0]
    # half: where fine-tuning helped/hurt most; half: random remaining errors (avoid only looking at extremes)
    pick = pd.concat([err.nsmallest(n // 4, "delta"), err.nlargest(n // 4, "delta")])
    pick = pd.concat([pick, err.drop(pick.index).sample(min(n - len(pick), len(err) - len(pick)), random_state=seed)])
    pick["auto_tags"] = [tags(r, y) for r, y in zip(pick.text, pick.finetuned)]
    pick["category"] = ""  # filled manually
    return pick.drop(columns="audio_filepath").round(3)


def curves(runs_dir, runs, out):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for name in runs:
        p = f"{runs_dir}/{name}/metrics.csv"
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        tr = d.dropna(subset=["train_loss"]) if "train_loss" in d else d.iloc[:0]
        va = d.dropna(subset=["val_loss"]) if "val_loss" in d else d.iloc[:0]
        ax[0].plot(tr.step, tr.train_loss.rolling(5, min_periods=1).mean(), label=name)
        ax[1].plot(va.step, va.val_loss, marker="o", label=name)
        ax[2].plot(va.step, 100 * va.val_wer, marker="o", label=name)
    ax[0].set(title="Train loss (5-pt smoothed)", xlabel="step")
    ax[1].set(title="Dev loss (held-out speakers)", xlabel="step")
    ax[2].set(title="Dev WER % - loss falls 10x, WER does not improve", xlabel="step")
    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def run(results, runs, runs_dir="/vol/runs", main="A1_full"):
    tab = ablation_table(results, runs, runs_dir)
    sub = subgroup_table(results, main)
    open(f"{results}/ablations.md", "w").write("## Ablations (eval = official valid, n per run below)\n\n" + tab.to_markdown(index=False)
                                               + f"\n\n## Subgroups: base vs {main} (normalized WER)\n\n" + sub.to_markdown(index=False) + "\n")
    error_sheet(results, main).to_csv(f"{results}/error_analysis.csv", index=False)
    curves(runs_dir, runs, f"{results}/training_curves.png")
    for p in glob.glob(f"{runs_dir}/*/run_info.json") + glob.glob(f"{runs_dir}/*/metrics.csv"):
        name = p.split("/")[-2]
        os.makedirs(f"{results}/{name}", exist_ok=True)
        shutil.copy(p, f"{results}/{name}/")  # small logs next to results, so the repo carries them
    print(open(f"{results}/ablations.md").read())
