"""IndicVoices-Marathi -> filtered, speaker-disjoint Canary manifests.

Splits
  eval  : official `valid` split (fixed, never used for any training decision)
  dev   : ~5% of speakers held out from the train pool (validation loss during training)
  train : sample from the rest of the train pool, excluding every eval speaker
  train_half : first half of the (shuffled) train set -> nested subset for the data-size ablation

Filtering is deliberately asymmetric: eval only drops rows that cannot be scored
(so the test set is not made artificially easy); train additionally drops rows whose
transcript is likely misaligned or that annotators flagged as bad.
"""
import ast
import io
import json
import os
import random

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

from src.text import DEVANAGARI_ONLY, clean

REPO = "ai4bharat/IndicVoices"
META = ["text", "duration", "speaker_id", "scenario", "task_name", "gender", "age_group", "district", "verification_report"]
BAD_FLAGS = ("unclear_audio", "wrong_language", "skipping_words", "incorrect_text_prompt")  # transcript may not match audio
SR = 16000


def reject_reason(r, strict):
    text, dur = clean(r["text"]), r["duration"]
    if not text:
        return "empty_text"
    if "<" in text or "[" in text:
        return "annotation_tag"  # e.g. <unintelligible>: not scorable, not learnable
    if not DEVANAGARI_ONLY.match(text):
        return "non_devanagari"
    if dur < 0.3 or dur > 30:
        return "duration_out_of_range"
    if not strict:
        return None
    cps = len(text.replace(" ", "")) / dur
    if cps > 25 or (dur > 5 and cps < 1):
        return "speaking_rate_outlier"  # likely truncated / misaligned transcript
    try:
        rep = ast.literal_eval(r["verification_report"] or "{}")
    except (ValueError, SyntaxError):
        rep = {}
    if any(rep.get(f) for f in BAD_FLAGS):
        return "annotator_flag"
    return None


def read_shard(path, split, strict):
    """Yield raw rows (audio still as encoded bytes) annotated with a rejection reason."""
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=256, columns=META + ["audio_filepath"]):
        for r in b.to_pylist():
            r["split"] = split
            r["reason"] = reject_reason(r, strict)
            yield r


def write_wav(audio, out):
    wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    sf.write(out, wav, SR)
    return len(wav) / SR


def manifest_line(r, path, dur):
    # Canary-2 AED prompt fields; transcript under both `answer` and `text` so either text_field works.
    return {"audio_filepath": path, "duration": round(dur, 3), "text": clean(r["text"]), "answer": clean(r["text"]),
            "source_lang": "mr", "target_lang": "mr", "pnc": "no", "taskname": "asr", "sampling_rate": SR,  # lets lhotse skip probing every file
            **{k: r[k] for k in ["speaker_id", "scenario", "task_name", "gender", "age_group", "district"]}}


def build(out_dir, train_shards, n_train, dev_frac, seed):
    rng = random.Random(seed)
    os.makedirs(f"{out_dir}/wavs", exist_ok=True)
    rows, keep = [], {}  # keep: split -> list of (row, audio)

    valid = hf_hub_download(REPO, "marathi/valid-00000-of-00001.parquet", repo_type="dataset")
    for r in read_shard(valid, "eval", strict=False):
        audio = r.pop("audio_filepath")
        rows.append(r)
        if r["reason"] is None:
            keep.setdefault("eval", []).append((r, audio))
    eval_speakers = {r["speaker_id"] for r in rows}

    for s in train_shards:
        p = hf_hub_download(REPO, f"marathi/train-{s:05d}-of-00072.parquet", repo_type="dataset")
        for r in read_shard(p, "train_pool", strict=True):
            audio = r.pop("audio_filepath")
            if r["reason"] is None and r["speaker_id"] in eval_speakers:
                r["reason"] = "speaker_in_eval"
            rows.append(r)
            if r["reason"] is None:
                keep.setdefault("pool", []).append((r, audio))

    # speaker-disjoint dev carve-out, then a seeded train sample
    pool_spk = sorted({r["speaker_id"] for r, _ in keep["pool"]})
    rng.shuffle(pool_spk)
    dev_spk = set(pool_spk[: max(1, int(len(pool_spk) * dev_frac))])
    keep["dev"] = [x for x in keep["pool"] if x[0]["speaker_id"] in dev_spk]
    rest = [x for x in keep.pop("pool") if x[0]["speaker_id"] not in dev_spk]
    rng.shuffle(rest)
    keep["train"] = rest[:n_train]

    manifests = {}
    for split, items in keep.items():
        lines = []
        for i, (r, audio) in enumerate(items):
            path = f"{out_dir}/wavs/{split}_{i:06d}.wav"
            lines.append(manifest_line(r, path, write_wav(audio, path)))
            r["used_in"] = split
        manifests[split] = lines
    manifests["train_half"] = manifests["train"][: len(manifests["train"]) // 2]  # nested subset
    for split, lines in manifests.items():
        with open(f"{out_dir}/{split}.json", "w") as f:
            f.writelines(json.dumps(l, ensure_ascii=False) + "\n" for l in lines)

    df = pd.DataFrame(rows).drop(columns=["verification_report"])
    df["reason"] = df["reason"].fillna("kept")
    df.to_parquet(f"{out_dir}/rows.parquet")  # one row per raw example, for profiling
    return {k: len(v) for k, v in manifests.items()}


def process_shard(shard, exclude_speakers, out_dir):
    """Scale-up path: one shard -> filtered manifest lines. Run many in parallel (one container each).
    Excludes eval *and* dev speakers, so eval/dev stay identical to the small runs and results are comparable."""
    os.makedirs(f"{out_dir}/wavs_large", exist_ok=True)
    p = hf_hub_download(REPO, f"marathi/train-{shard:05d}-of-00072.parquet", repo_type="dataset")
    lines, rows = [], []
    for i, r in enumerate(read_shard(p, "train_large_pool", strict=True)):
        audio = r.pop("audio_filepath")
        if r["reason"] is None and r["speaker_id"] in exclude_speakers:
            r["reason"] = "speaker_in_eval_or_dev"
        r.pop("verification_report")
        rows.append(r)
        if r["reason"] is None:
            path = f"{out_dir}/wavs_large/s{shard:02d}_{i:06d}.wav"
            lines.append(manifest_line(r, path, write_wav(audio, path)))
    return lines, rows
