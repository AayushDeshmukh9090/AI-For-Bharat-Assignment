"""Dataset profile: the numbers for the README + two plots. Reads rows.parquet and the manifests from data.py."""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SPLITS = ["train", "train_half", "dev", "eval"]


def load_manifest(path):
    return pd.DataFrame([json.loads(l) for l in open(path)])


def text_stats(m, train_vocab=None):
    words = m.text.str.split()
    vocab = Counter(w for ws in words for w in ws)
    s = {"words_per_utt_median": float(words.str.len().median()), "chars_per_utt_median": float(m.text.str.len().median()),
         "vocab_size": len(vocab), "total_words": int(sum(vocab.values()))}
    if train_vocab is not None:  # token-level OOV of this split w.r.t. training vocabulary
        s["oov_rate_vs_train_%"] = round(100 * sum(c for w, c in vocab.items() if w not in train_vocab) / s["total_words"], 2)
    return s


def run(data_dir, out_dir):
    rows = pd.read_parquet(f"{data_dir}/rows.parquet")
    man = {s: load_manifest(f"{data_dir}/{s}.json") for s in SPLITS}
    train_vocab = {w for t in man["train"].text for w in t.split()}

    prof = {"raw": {}, "splits": {}}
    for src, g in rows.groupby("split"):  # raw source = eval (official valid) / train_pool (sampled shards)
        prof["raw"][src] = {"raw_examples": len(g), "valid_examples": int((g.reason == "kept").sum()),
                            "removed_examples": int((g.reason != "kept").sum()),
                            "removed_by_reason": g.reason[g.reason != "kept"].value_counts().to_dict(),
                            "raw_hours": round(g.duration.sum() / 3600, 2)}
    for s, m in man.items():
        prof["splits"][s] = {"utterances": len(m), "hours": round(m.duration.sum() / 3600, 2),
                             "duration_s": {"median": round(m.duration.median(), 2), "min": round(m.duration.min(), 2), "max": round(m.duration.max(), 2)},
                             "speakers": m.speaker_id.nunique(), "utts_per_speaker_median": float(m.speaker_id.value_counts().median()),
                             "scenario": m.scenario.value_counts().to_dict(), "gender": m.gender.value_counts().to_dict(),
                             "text": text_stats(m, None if s.startswith("train") else train_vocab)}
    spk = {s: set(m.speaker_id) for s, m in man.items()}
    prof["speaker_overlap"] = {"train∩eval": len(spk["train"] & spk["eval"]), "train∩dev": len(spk["train"] & spk["dev"]),
                               "dev∩eval": len(spk["dev"] & spk["eval"])}
    json.dump(prof, open(f"{out_dir}/profile.json", "w"), indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for s in ["train", "eval"]:
        ax[0].hist(man[s].duration, bins=60, alpha=0.6, label=f"{s} (n={len(man[s])})", density=True)
        ax[1].hist(man[s].speaker_id.value_counts(), bins=40, alpha=0.6, label=s)
    ax[0].set(title="Utterance duration", xlabel="seconds", ylabel="density")
    ax[1].set(title="Utterances per speaker", xlabel="utterances", ylabel="speakers")
    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/data_profile.png", dpi=130)
    return prof
