# Fine-tuning Bodhan Indic-Transcribe-core on Marathi (IndicVoices)

End-to-end fine-tuning of `bodhan-ai/indic-transcribe-core` (1.2B, Canary-style) on IndicVoices-Marathi, run on Modal H100s: a baseline, five fine-tuning ablations, a subgroup analysis, and an error analysis.

**Short version:** the pipeline runs cleanly start to finish, but none of the five fine-tuning variants actually beat the untouched base model. Base WER/CER is 12.33 / 4.24; the best fine-tuned run (A4, trained on 136 hours of audio) lands at 12.83 / 4.42. That's not a fluke — the ablations explain why, and I think the explanation is more useful than a number that happened to go up. Dev loss falls tenfold in the first hundred steps while dev WER climbs, which means most of what the model is "learning" is IndicVoices' own transcription style, not better speech recognition. The damage concentrates on short conversational replies (हां turning into हो, अच्छा into ओके), and it shrinks the more distinct data you throw at it.

Artifacts (checkpoints, full logs, all predictions) are on the Google Drive link in the submission email — the best checkpoint is `A4_large.nemo`.

## 0. Why Core, why IndicVoices — the reasoning before any code got written

Before touching the training pipeline I spent a chunk of the morning on two decisions that are easy to get wrong in a hurry: which of the two Bodhan ASR models to use, and which dataset. Both turned out to have a somewhat non-obvious right answer, so I'm writing out how I actually got there rather than just stating the conclusion.

**Picking the model.** Bodhan ships two ASR models — Core and Flex. My first instinct was to just look at the leaderboard: Core's Marathi OIWER on the Voice of India benchmark is 5.7, Flex's is 6.6, so Core wins, done. I talked myself out of that pretty quickly — a one-point gap on a benchmark neither model was fine-tuned for isn't a real basis for a decision, it's just noise dressed up as a number.

So I reframed the question. It isn't "which model is more accurate in general," it's "which model's output convention matches what my training data's ground truth actually looks like." Flex's whole pitch is that it can keep English loanwords in Latin script (mode="mixed") instead of forcing everything into Devanagari. That's a genuinely useful feature — if the training data ever demonstrates it. If it doesn't, fine-tuning on Flex's mixed-script mode is fine-tuning a capability nobody is going to supervise.

This got more confusing before it got clearer. Bodhan's own "Voice of India" benchmark writeup (co-authored with Mitesh Khapra) argues, with a worked example, that all-native transliteration undersells how Indians actually speak — their example keeps "वह doctor के पास गया" with "doctor" in Latin script, on the reasoning that forcing it into Devanagari loses information. Reading that pushed me toward Flex for a bit.

What settled it was just looking at the data instead of arguing about it in the abstract. I pulled about two dozen IndicVoices Marathi rows from the HF dataset viewer and checked the verbatim/normalized columns for every English loanword I could find: price → प्राइस, discount → डिस्काउंट, variety → व्हरायटीज, crab → क्रॅब, English → इंग्लिश, medium → मिडियम, dry → ड्राय. Every single one, fully transliterated into Devanagari. Zero Latin-script tokens anywhere in the sample. IndicVoices' annotation convention is native-script-only, full stop — whatever Bodhan's benchmark essay argues about real-world speech, it doesn't describe how this particular dataset was annotated. Core's output convention matches that ground truth exactly; Flex's differentiating feature would be training against data that never shows it what it's supposed to do. Core was the right call, and for a reason grounded in the actual training data rather than a leaderboard number.

Caveat I'm carrying forward honestly: twenty-four rows is a spot check, not a census. I wrote a small script (`check_script_convention.py`, not part of the main pipeline since it was a one-off sanity check) to verify the convention against a larger sample before committing to the main run, as cheap insurance against a rare mixed-script minority — proper nouns, mostly — hiding outside what I happened to sample.

One more thing worth flagging honestly: Bodhan hasn't published what Core was pretrained on. Third-party reporting mentions "1.3 million hours of audio, combining weak supervision, synthetic speech and human-labelled data" with no dataset names given. Given that Core and IndicVoices come out of the same AI4Bharat/Bodhan lineage, it's plausible (not confirmed) that Core saw some IndicVoices-like data during pretraining already. I don't have a way to rule that out, so I'm treating any base-vs-fine-tuned gap in this report as a lower bound on genuine adaptation, not a clean measurement — better to say that up front than to bury it in a footnote.

**Picking the dataset.** The realistic options were IndicVoices and Shrutilipi, both Marathi, both CC-BY-4.0, both from the AI4Bharat/Khapra group. Shrutilipi is All India Radio news — read, clean, single-speaker-per-clip, no official validation split and no speaker metadata. IndicVoices is much bigger (about 300k train / 3.7k valid on the official split) and comes with real metadata: speaker id, gender, age group, district, state, scenario, task name. Composition-wise it's 8% read, 76% extempore, 15% conversational.

I went with IndicVoices for three reasons. First, it already has an official train/valid split, which removes a chunk of engineering work — though "removes the work of building a split" is not the same as "removes the work of checking the split is actually clean," which is exactly the leak I found later in §4. Second, the rich metadata is what makes a real subgroup analysis possible instead of a single aggregate WER number. Third, and probably most important: spontaneous, extempore speech is the genuinely hard case for Indic ASR — disfluencies, code-mixing, dialectal pronunciation — and it's a far more honest test of a model than clean broadcast news. Shrutilipi would have been the easier dataset to get a nice-looking number on, which is exactly why it felt like the wrong choice for something meant to demonstrate judgement rather than produce a leaderboard entry.

Practical constraint: 300k training utterances is far more than a one-day budget supports, so the plan going in was a subsample in the low thousands, kept speaker-disjoint from a similarly sized eval slice pulled from the official valid split — with the disjointness actually checked against `speaker_id`, not assumed. (It turned out not to be disjoint out of the box — see §4.)

**The hypothesis I fixed before running anything.** Fine-tuning on IndicVoices — which is over 90% spontaneous speech — should help spontaneous speech (extempore + conversational) more than it helps read speech, because spontaneous speech is where Indic ASR genuinely struggles and is plausibly the register the base model has seen the least of during pretraining. I wrote this down before the main training run, specifically so I couldn't quietly reshape the hypothesis around whatever the results turned out to be. §7 reports what actually happened to it — the short version is it held up in relative terms but not in absolute ones, which is itself a more interesting finding than a clean confirmation would have been.

## 1. Objective

Get a fine-tuning run of a Bodhan ASR model working end to end on Marathi, and be honest about what actually happened. That meant a fixed eval set nothing in training gets to see, speakers kept apart between splits, the base model scored on the exact same set as the fine-tuned ones, and enough analysis to say *where* a change comes from rather than just reporting one number and moving on. I had about a day for this, so everything below leans toward "small, correct, and reproducible" over "big and impressive-looking."

## 2. Model: `indic-transcribe-core`, fine-tuned through NeMo

| | |
|---|---|
| Architecture | FastConformer encoder (32L, 811M) + Transformer decoder (24L, 419M), Canary-1b-v2 lineage |
| Output | Native script only, prompt-conditioned (`source_lang`, `target_lang`, `pnc`) |
| Why Core, not Flex | see §0 — IndicVoices references are native-script only, so Core's output convention is the one that actually matches the training signal |
| Why NeMo, not the HF class | The HF `trust_remote_code` class on the model page is inference-only. Training needs the shipped `.nemo` checkpoint, loaded through NeMo's own `EncDecMultiTaskModel`, with its lhotse dataloader and Lightning training loop |

## 3. Dataset: `ai4bharat/IndicVoices`, Marathi config

Reasoning for IndicVoices over Shrutilipi is in §0. Licence is CC-BY-4.0.

**Budget decision:** the Marathi train split is spread across 72 parquet shards (roughly 470k utterances total). I used 3 shards, spaced out across the split (numbers 3, 27, and 51, so as not to accidentally sample from one contiguous recording batch) as the train pool — enough for a 6k-utterance run, and small enough that download and preprocessing take minutes rather than hours.

### Profile (`results/profile.json`, `results/data_profile.png`)

| | eval (official `valid`) | train pool (3 shards) |
|---|---|---|
| raw examples | 3,679 | 12,470 |
| removed | 124 | 4,211 |
| valid examples | **3,555** | 8,259 |
| raw hours | 6.15 | 23.04 |

| split | utts | hours | median / min / max dur (s) | speakers | read / extempore / conversation |
|---|---|---|---|---|---|
| train | 6,000 | 10.92 | 5.78 / 0.30 / 25.7 | 438 | 595 / 3,054 / 2,351 |
| train_half | 3,000 | 5.40 | 5.63 / 0.30 / 25.7 | 397 | 296 / 1,528 / 1,176 |
| dev | 275 | 0.61 | 7.17 / 0.35 / 23.0 | 23 | 37 / 207 / 31 |
| eval | 3,555 | 6.03 | 5.06 / 0.30 / 28.9 | 435 | 204 / 1,273 / 2,078 |

For the data-scale ablation (A4) I went further and pulled 24 shards instead of 3 (every third one, so still spread across the split): 99,754 raw rows, 69,059 of them valid after filtering, 136.1 hours, 1,077 unique speakers (`results/profile_large.json`). Of what got dropped, 28,603 rows were removed for having a speaker who also shows up in eval or dev, 1,288 for annotation tags, 476 for duration, 306 flagged by the annotators themselves, and 22 as speaking-rate outliers. Each shard was processed in its own CPU container so this whole step took a few minutes rather than tying up one machine.

On the text side: a median of 11–12 words per utterance, a training vocabulary of 15.8k word types, and — worth calling out — 14% of eval word tokens don't appear anywhere in the 6k-utterance training set. So any improvement has to come from genuine sub-word generalisation, not from the model having simply memorised the words it needs. Speaker overlap across the splits I actually train and evaluate on is zero in every direction (train∩eval, train∩dev, dev∩eval all empty).

![profile](results/data_profile.png)

## 4. Data preparation (`src/data.py`)

The first thing I checked, before writing any filtering logic, was whether the official `valid` split is actually disjoint from `train` by speaker. It isn't. 3,885 of the 12,470 train-pool rows I pulled — 31% — come from speakers who also appear in `valid`. Training on those would leak speaker identity straight into the evaluation set, so they get dropped from training entirely (eval itself is left untouched — I'm not going to clean up the test set to make my own numbers look better).

Filtering is deliberately asymmetric between eval and train. Eval only loses rows that literally can't be scored. Train additionally loses rows where the transcript is probably just wrong.

| filter | eval | train pool | why |
|---|---|---|---|
| `annotation_tag` (`<unintelligible>`-style tags in text) | 59 | 162 | not scorable, not learnable — these tags are the only Latin text anywhere in the corpus |
| duration outside 0.3–30 s | 65 | 98 | sub-0.3 s clips have no speech in them; the model trains on clips up to 30 s |
| non-Devanagari characters | 0 | 0 | sanity check on the native-script assumption from §0 — it held across the full split |
| speaking-rate outlier (>25 chars/s, or <1 char/s on clips over 5 s) | – | 5 | usually a sign the transcript is truncated or misaligned |
| annotator flags (`unclear_audio`, `wrong_language`, `skipping_words`, `incorrect_text_prompt`) | – | 61 | the annotators' own verification report says the audio and text may not match |
| speaker also appears in eval | – | 3,885 | leakage, described above |

A few other things worth noting about the pipeline:

- **Dev is speaker-disjoint too.** 5% of train-pool speakers (23 of them, 275 utterances) are held out for validation loss, and `valid` never factors into any training decision.
- **The half-data ablation is a true subset.** `train_half` is the first 3,000 rows of the seeded, shuffled `train` set, so A3 sees a strict subset of what A1 sees rather than an independently sampled 3k.
- **The manifest schema isn't the usual `{audio_filepath, text}`.** Canary-2 is a prompted multi-task model, so each manifest line also carries `source_lang=mr`, `target_lang=mr`, `pnc=no`, `taskname=asr`, and the transcript under `answer` (kept under `text` too, for convenience). `pnc=no` because IndicVoices references don't carry punctuation.
- Audio comes out of the parquet files as raw bytes, gets decoded, downmixed to mono, and resampled to 16 kHz WAV.

## 5. Training (`src/train.py`, `configs/train.yaml`)

Training uses NeMo's own model class, its lhotse dataloader, and its optimiser setup, but I run the loop myself rather than going through `speech_to_text_finetune.py`, mainly because it lets each ablation just be a dict override on top of one shared config rather than five separate YAML files.

| setting | value | why |
|---|---|---|
| optimiser | AdamW, lr 1e-5, wd 1e-3, betas (0.9, 0.98) | strong pretrained model + small in-domain dataset, so a small step size to avoid wrecking what it already knows |
| schedule | 50 warmup steps, cosine down to lr/20 | fairly standard for this kind of fine-tune |
| steps | 500 (~5 passes over 6k utterances) | comfortably fits the time budget; validating every 100 steps shows whether more would even help |
| batching | lhotse dynamic batching, 360 s of audio per batch, no bucketing | kept simple on purpose — the 2D bucketing in the original Canary recipe is tuned for their own data mix, not mine |
| precision | bf16-mixed on a single H100 | |
| grad clip | 1.0 | |
| logging | CSV of train loss / val loss / lr per step, plus `run_info.json` for runtime, peak GPU memory, and trainable parameter count | |

### Ablations (one H100 each, run in parallel)

| run | change vs A1 | question it's meant to answer |
|---|---|---|
| A0 base | no fine-tuning at all | the reference point everything else is measured against |
| A1_full | full fine-tune, 6k utterances, 500 steps | does in-domain fine-tuning help at all? |
| A2_frozen_enc | encoder frozen (419M of 1.2B params trainable) | is any gain coming from the acoustic side or the language/decoder side? |
| A3_half_data | 3k utterances (a nested subset of A1's data), 250 steps to match the epoch count | how much does raw data volume matter at this scale? |
| A4_large | 24 shards, 69k utterances / 136 h, time-boxed to 22 minutes | does scaling up the data help beyond what A1 gets, at a roughly similar compute budget? |
| A5_top2 | only the last 2 encoder layers, last 2 decoder layers, and the output head trainable | does restricting capacity prevent the kind of forgetting A1/A2 show? |

One detail that matters for reading the dev-loss curves later: `trainer.validate()` runs once before `fit()` starts, so every curve in §7 begins at the base model's own dev loss — nothing is cherry-picked from partway through training.

### Smoke test (`src/smoke_test.py`)

Before spending real GPU time, the smoke test runs the exact same `train.run` function the real ablations use, on 16 utterances for 3 steps, then saves the checkpoint, reloads it fresh from disk, checks that the weights actually moved, and runs inference on 4 clips from that reloaded checkpoint. That's load → tokenizer → collate → forward → loss → backward → optimizer step → checkpoint → reload → inference, all exercised in under two minutes, using production code rather than a simplified stand-in.

## 6. Evaluation (`src/evaluate.py`, `src/text.py`)

The eval set is the official `valid` split, minus the handful of rows that genuinely can't be scored: 3,555 utterances, 6.0 hours, 435 speakers, none of them seen during training, and the exact same set used across every run. WER and CER come from jiwer, computed two ways: a "raw" pass that only does NFC normalisation and whitespace collapsing, and a "normalized" pass that also strips punctuation (including the Marathi danda `।`), zero-width joiners (common in Marathi — प्रिंटस्‌ has one), and casing. The gap between the two tells you how much of a WER number is really about formatting rather than recognition. Subgroup breakdowns (register, scenario, duration bucket, gender) are computed on the normalized scoring. Fine-tuned models are always evaluated from their reloaded `.nemo` file rather than the in-memory model still sitting in the training process — otherwise a checkpoint could look fine at eval time and just not actually be usable.

## 7. Results

All numbers below come from the same 3,555-utterance eval set, and every fine-tuned model is scored from its saved-and-reloaded checkpoint. WER/CER are in percent; "CER no-space" is CER computed after stripping all spaces, i.e. it ignores word-segmentation disagreements.

| run           |   WER raw |   CER raw |   WER norm |   CER norm |   CER no-space |   WER read |   WER spont. | trainable M   | train min   | peak GB   |
|:--------------|----------:|----------:|-----------:|-----------:|---------------:|-----------:|-------------:|:--------------|:------------|:----------|
| base          |     12.33 |      4.24 |      12.33 |       4.24 |           4.39 |       5.95 |        12.81 | -             | -           | -         |
| A1_full       |     13.11 |      4.56 |      13.11 |       4.56 |           4.71 |       7.19 |        13.55 | 1221          | 10.2        | 61.5      |
| A2_frozen_enc |     13.99 |      5.02 |      13.99 |       5.02 |           5.15 |       8.2  |        14.42 | 410           | 11.1        | 17.7      |
| A3_half_data  |     13.38 |      4.68 |      13.38 |       4.68 |           4.8  |       8.01 |        13.78 | 1221          | 7.8         | 57.3      |
| A4_large      |     12.83 |      4.42 |      12.83 |       4.42 |           4.58 |       7.83 |        13.2  | 1221          | 26.6        | 54.9      |
| A5_top2       |     14.37 |      5.06 |      14.37 |       5.06 |           5.14 |       8.35 |        14.82 | 84            | 13.8        | 12.8      |

![curves](results/training_curves.png)

### What the ablations are actually telling me

Every single run shows the same pattern: dev loss drops from 1.34 down to around 0.12 within the first 100 steps, and dev WER goes up over that same stretch (from 8.9% to somewhere between 9.3% and 9.9%, depending on the run), and eval WER follows the same direction. A tenfold drop in loss with zero recognition improvement isn't really consistent with the model getting better at speech recognition — it's much more consistent with the model rapidly fitting IndicVoices' own transcription conventions and prompt formatting under teacher forcing, which a model this strong was probably already getting mostly right beforehand. If I'd picked checkpoints by dev loss the way you normally would, I'd have picked the worst ones on every axis that actually matters. That's probably the single most useful thing this project surfaced.

Data volume turned out to be the strongest lever by a decent margin: 13.38 on 3k utterances, 13.11 on 6k, 12.83 on 69k/136 hours — the gap to the base model shrinks at every step up. A4 even fully closed the short-clip regression that shows up in every other run (19.84 vs the base's 19.7), and it did that despite being capped at 609 steps — under half an epoch over that much data. So at a fixed compute budget, more distinct data clearly beats more passes over the same small set.

Freezing the encoder (A2) was the worst option among the "reasonable" variants at 13.99. My guess going in was that if the base model's weakness were acoustic, freezing the encoder and only touching the decoder should help. It did the opposite — with only the decoder trainable, all the adaptation pressure gets funneled into the language-model side, which is exactly the part that was already breaking (the backchannel substitutions and the odd Hindi intrusion described in §8). One genuinely useful side effect: freezing the encoder cut peak memory from 61.5 GB down to 17.7 GB, which matters if you're ever doing this on smaller hardware.

Restricting to just the top two layers (A5) was worse still, at 14.37, and this is the one that actually surprised me — I expected limiting how much of the network could move would limit how much damage got done. Instead it concentrated all the adaptation into the output side even harder than A2 did, and the short-clip bucket fell apart completely (33.05 WER against a base of 19.7). Lined up, the five runs go A4 < A1 < A3 < A2 < A5 — full-network fine-tuning with more distinct data caused the least damage, and adaptation squeezed into just the output layers caused the most. If I were to chase this further, the next thing to try isn't fewer trainable layers, it's a lower learning rate or LoRA spread across the whole network with early stopping on dev WER instead of dev loss.

One thing that made basically no difference: raw versus normalized scoring — identical to two decimal places on every run. Under the `pnc=no` prompt the model never emits punctuation, digits, or Latin script in the first place, so there was nothing for normalization to clean up. The space-insensitive CER runs about 0.15 points above regular CER, which tells me a small but real slice of the errors are just word-boundary conventions (अशाप्रकारचे vs अशा प्रकारचे) rather than actual recognition mistakes.

### Going back to the hypothesis from §0

The hypothesis, stated before any training happened: fine-tuning on IndicVoices — 91% spontaneous speech — should help spontaneous speech more than read speech.

| axis       | group        |    n |   WER base |   WER A4_large |   ΔWER |
|:-----------|:-------------|-----:|-----------:|---------------:|-------:|
| register   | read         |  204 |       5.95 |           7.83 |   1.88 |
| register   | spontaneous  | 3351 |      12.81 |          13.2  |   0.39 |
| scenario   | Conversation | 2078 |      16.07 |          16.38 |   0.31 |
| scenario   | Extempore    | 1273 |       9.85 |          10.32 |   0.47 |
| scenario   | Read         |  204 |       5.95 |           7.83 |   1.88 |
| dur_bucket | long >10s    |  767 |      11.06 |          11.63 |   0.57 |
| dur_bucket | medium 2-10s | 1679 |      12.75 |          13.23 |   0.48 |
| dur_bucket | short <2s    | 1109 |      19.7  |          19.84 |   0.14 |
| gender     | Female       | 1846 |      11.33 |          11.86 |   0.53 |
| gender     | Male         | 1707 |      13.46 |          13.93 |   0.47 |

Nothing improved in absolute terms — that much is just consistent with everything above. But the pattern of *which* subgroup got hurt the least is actually the interesting part, and it does line up with the hypothesis in a relative sense: read speech took the biggest hit (+1.88 WER) and spontaneous speech the smallest (+0.39). Fine-tuning pulled the model toward the register it was trained on, and read speech — the register barely represented in that training data — paid the price for being out of distribution. So: directionally supported, not confirmed outright. Worth flagging that read only has 204 examples in eval, so I wouldn't lean too hard on that specific number. Conversation, which is the single hardest scenario at 16% base WER, was also the most stable one (+0.31), which is a small bit of good news buried in an otherwise not-great result.

And the caveat from §0 still applies here: since Bodhan hasn't said what Core was pretrained on, and it shares an institutional lineage with IndicVoices, it's entirely possible the base model has already seen data like this. That would explain why it's already doing reasonably well and why fine-tuning has "nothing much left to learn" except conventions. I'd treat every delta in this report as a floor on what real fine-tuning could achieve on genuinely unseen data, not a ceiling.

## 8. Error analysis

`results/error_analysis.csv` has 50 hand-reviewed utterances: the 12 where fine-tuning helped the most, the 12 where it hurt the most, and 26 more picked at random from the remaining errors, so the sample isn't just cherry-picked good or bad cases. Each row has the reference, base output, fine-tuned output, per-utterance WER, and a few auto-generated tags (`very short ref`, `deletions`, `insertions/hallucination`, `word order`, `non-Devanagari output`) that I used as a starting point — the actual categories below came from reading through the examples, not from a taxonomy I'd decided on beforehand.

Looking at a random sample of base-model errors, most of them aren't really recognition failures at all — they're convention mismatches:

| category | example (reference → base) |
|---|---|
| spelling variant | किंमतींवर → किमतींवर, हा → हां |
| dialect written as spoken | तुमी → तुम्ही (the reference keeps the spoken form; the model quietly standardises it) |
| word segmentation | अशाप्रकारचे → अशा प्रकारचे; बघितलेली → बघितली ती |
| disfluency / false start | the reference keeps mid-sentence fragments (…महिलांच्या **साद** महिलां…, a stray **र**); the model just smooths them out |
| genuine misrecognition | शॉपवर → वापर (a noisy conversational clip) |

Also worth noting: the base model, under the `pnc=no` prompt, never outputs punctuation, digits, or Latin script — which is exactly why raw and normalized WER come out identical, and why the space-insensitive CER barely moves from plain CER.

What fine-tuning actually broke shows up most clearly on short conversational backchannels — the "hmm," "okay," "right" of Marathi conversation. On clips under 2 seconds, 55 utterances got worse after fine-tuning and 36 got better, and because the references here are often just one or two words, each mistake counts for a lot: WER on this bucket jumps from 19.7% to somewhere in the high 20s depending on the run. The pattern is consistent — the fine-tuned decoder's language-model prior drifts toward whatever backchannel showed up most often in the small training set:

| reference | base | fine-tuned (A3) |
|---|---|---|
| हां | हां | हो |
| अच्छा | अच्छा | ओके |
| ठीक आहे | ठीक आहे | हो का |
| हां गोळ्या आहेत ना त्या | (correct) | हां गोव्या **इतना** ते ← Hindi intrusion |
| हॅलो नितीन बोलतो ना | (correct) | …बोलता ना ← verb-ending drift |
| चालेल चालेल चालेल | (correct) | चाल चाल चाल ← truncation |

With so little acoustic signal to go on in a sub-2-second clip, the model leans on its prior — and 6,000 utterances turned out to be plenty to shift that prior, but not nearly enough to actually improve the underlying acoustics. I tried the obvious fix for this (A5, restrict which layers can move) and it made the problem noticeably worse — 33.05 WER on this same bucket. What actually worked was more distinct training data: A4 fully recovered this bucket back to base-level performance. The other levers worth trying are a lower learning rate and early stopping on dev WER rather than dev loss.

## 9. Problems encountered

Roughly in the order I ran into them:

1. **No documented HF training path.** The model card's "no NeMo needed" line is about inference. Fine-tuning needs the `.nemo` checkpoint and NeMo 2.7.
2. **Custom tokenizer registration.** The checkpoint's config names `nemo.collections.common.tokenizers.canary_multilingual_tokenizer.CanaryMultilingualTokenizer`, which a stock NeMo install doesn't have. My first attempt just did a plain `import` of the file, which isn't enough — the class has to be registered under NeMo's own module path before `restore_from` will accept it. I caught this by reading the repo's own `nemo/load_nemo.py` before touching a GPU, and now call its `register_tokenizer()` helper directly.
3. **Gated access, wrong account.** The Hugging Face gate is per-account, and the token sitting in the Modal secret belonged to an account that hadn't actually been granted access yet — a 403 on the `.nemo` files specifically, while the dataset (a separate gate) worked fine. Tracked it down by printing `whoami()` from inside the container.
4. **Speaker leakage in the official split** — 31% of the sampled train rows. Covered in §4.
5. **The manifest schema isn't the standard one.** Canary needs `source_lang`/`target_lang`/`pnc` plus the transcript under `answer`, not the usual `{audio_filepath, text}` pair. Built this into preprocessing from the start rather than discovering it at training time.
6. **A silent "training in eval mode" bug — probably the most instructive thing that went wrong.** The first launch of A1 through A3 printed `0 Modules in train mode, 1460 Modules in eval mode` in the Lightning startup summary. `model.load()` returns the model in `.eval()` mode (correct for inference), and it turns out Lightning 2.2+ *preserves* whatever mode a module is already in when `fit()` starts, rather than forcing `.train()` the way I assumed it would. Those runs would have trained with SpecAugment and dropout both silently disabled — no crash, no warning, just a quietly different recipe than intended. I caught it about ten minutes in from the summary line, added an explicit `.train()` call, and relaunched. The smoke test hadn't caught this because loss still goes down fine even in eval mode — the lesson here is to actually read what the training summary prints, not just watch the loss curve.
7. **A scary-looking log line that turned out to be harmless.** Every restore prints `ERROR:hydra.utils: Error getting class at ...get_nemo_transformer`, immediately followed by a successful restore message. The decoder's config names a factory function where hydra expects a class name, and NeMo just falls back gracefully. Confirmed harmless by checking the actual inference output, then ignored.
8. **A bug in my own smoke-test assertion.** I checked `model.decoder` weights before/after training, but Canary's decoder attribute is actually called `transf_decoder`. This crashed the test *after* training, saving, and reloading had all already succeeded — so the underlying pipeline was fine, my check of it wasn't. Now it compares every parameter tensor (1,826 of them changed after 3 steps).
9. **An orchestration bug plus a Modal quirk.** `modal run --detach` only keeps the *last* function you triggered alive if the local process disconnects, so fanning five runs out from the local entrypoint directly would have silently lost runs. Moved the fan-out into a remote function instead. Separately, an early version of that function passed the run's *name* into a spot where `train.run` expected the override dict, which threw `TypeError: 'str' object is not a mapping` immediately — annoying, but at least it failed loudly and fast rather than quietly training the wrong config.
10. **Training silently stalled at 0% GPU utilization — a data-loading trap.** After relaunching, three of four containers just sat there with only the model weights (5.4 GB) loaded and no GPU activity. I attached `py-spy` to one of the live containers (`modal container exec` plus a stack dump) and found the main process stuck inside `soundfile.info()`, called from NeMo's lhotse adapter on its way to building a `Recording` object. Turns out that when a manifest line doesn't specify `sampling_rate`, NeMo opens *every single audio file* just to read its header before training can start. With a 10k-item shuffle buffer, that's thousands of file opens over a network volume before the first batch even runs — about 15 minutes for 6k files, and it would have been closer to 25 for A4's 69k. The fix was one field: write `sampling_rate: 16000` into every manifest line, so NeMo can build the `Recording` from the manifest alone. A3 happened to finish before hitting this because its 3k files fit inside the window; A1, A2, and A4 all needed restarting with the fixed manifests. The exact same trap hit evaluation too — `model.transcribe()` on a plain list of file paths writes its own temporary manifest without a sampling rate, so the base-model eval sat idle for about 25 minutes probing 3,555 files. Fixed by passing our own manifest directly into `transcribe()` instead.
11. **Compute scale versus the size of the actual dataset.** 6,000 utterances is small next to the roughly 470,000 available. I looked into multi-GPU training but ruled it out — Lightning's DDP launcher wants to re-exec the script itself, which doesn't play well inside a Modal function, and debugging that wasn't a good use of the remaining time. Instead A4 scales up the *data* by about 8x (24 shards instead of 3), prepared across 24 parallel CPU containers in a few minutes, then trained on a single H100 with more loader workers and a hard wall-clock cap of 22 minutes. At a fixed compute budget, more distinct data clearly beat more epochs over the same small set — see §7.

## 10. What I'd do next

1. **Score with something orthography-aware, not plain WER/CER.** A lot of what I'm counting as a base-model "error" is really a legitimate spelling variant (किंमत/किमत), a dialect form written the way it was actually spoken (तुमी vs तुम्ही), or a segmentation choice (अशाप्रकारचे vs अशा प्रकारचे). Something like Bodhan's own OIWER metric would separate genuine recognition gains from convention-matching, and that distinction matters more than any single number in this report.
2. **Do the A4 recipe properly, with more GPUs and more time.** Train across multiple GPUs (torchrun inside the container, or Modal's multi-node setup) on all 72 shards, use NeMo's 2D bucketing to cut down on padding waste, and — this one matters — pick checkpoints by dev WER, not dev loss. The A3 curve is a clean illustration of why those two disagree here.
3. **Go after the actual weak points directly.** Conversation (16% base WER) and clips under 2 seconds (about 20% WER) are where the model struggles the most. Up-weighting those during sampling, or giving the model more context around short backchannels like हां/हो, targets the real failure mode instead of just optimizing an aggregate.

## Reproduce

```bash
pip install modal && modal setup
modal secret create huggingface HF_TOKEN=<token with access to the gated model + dataset>
modal run modal_app.py --stage prepare        # ~15 min, CPU: download, filter, manifests, profile
modal run modal_app.py --stage prepare_large  # 24 shards, one CPU container each (for A4)
modal run modal_app.py --stage smoke          # ~10 min, 1×H100
modal run modal_app.py --stage ablations      # base eval + A1..A5 in parallel, 1×H100 each
# (--only A4_large to launch a single run)
modal run modal_app.py --stage analyze        # tables, curves, error sheet
modal volume get marathi-asr results ./       # pull results
```

## Layout

```
configs/train.yaml      all hyper-parameters + ablation overrides
modal_app.py            the only entrypoint; images, volume, stages
src/data.py             IndicVoices -> filtered, speaker-disjoint Canary manifests
src/profile_data.py     dataset statistics + plots
src/model.py            load .nemo (custom tokenizer registration), transcribe
src/train.py            NeMo/Lightning fine-tuning, ablation = config override
src/smoke_test.py       3-step train -> save -> reload -> infer
src/evaluate.py         predictions.json + metrics.json (raw/normalized, subgroups)
src/analyze.py          ablation table, subgroup deltas, curves, error sheet
src/text.py             normalisation + WER/CER (self-test: python -m src.text)
results/                everything small; checkpoints/logs are on Google Drive
```