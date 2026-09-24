"""Transcribe the fixed eval manifest -> predictions.json + metrics.json (raw & normalised, plus subgroups)."""
import json
import os
import time

import pandas as pd

from src.model import transcribe
from src.text import wer_cer

# Buckets chosen from the data profile: 25% of eval clips are < 1.3 s one-word backchannels ("हां").
DUR_BINS, DUR_LABELS = [0, 2, 10, 31], ["short <2s", "medium 2-10s", "long >10s"]


def subgroups(df):
    df = df.assign(register=df.scenario.map(lambda s: "read" if s == "Read" else "spontaneous"),
                   dur_bucket=pd.cut(df.duration, DUR_BINS, labels=DUR_LABELS).astype(str))
    return {col: {k: wer_cer(g.text, g.hyp, norm=True) for k, g in df.groupby(col)}
            for col in ["register", "scenario", "dur_bucket", "gender"]}


def run(m, manifest, out_dir, c):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame([json.loads(l) for l in open(manifest)])
    start = time.time()
    df["hyp"] = transcribe(m, manifest, c)
    secs = time.time() - start

    keep = ["audio_filepath", "text", "hyp", "duration", "speaker_id", "scenario", "gender"]
    df[keep].to_json(f"{out_dir}/predictions.json", orient="records", force_ascii=False, indent=1)
    metrics = {"raw": wer_cer(df.text, df.hyp), "normalized": wer_cer(df.text, df.hyp, norm=True),
               "subgroups": subgroups(df), "rtf": round(secs / df.duration.sum(), 4)}
    json.dump(metrics, open(f"{out_dir}/metrics.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: metrics[k] for k in ["raw", "normalized", "rtf"]}))
    return metrics
