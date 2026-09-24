"""Text normalisation + WER/CER. Shared by data prep and evaluation so both sides score identically."""
import re
import string
import unicodedata

import jiwer

_PUNCT = re.compile("[" + re.escape(string.punctuation) + "।॥“”‘’…–—]")
_ZERO_WIDTH = re.compile("[​‌‍﻿]")
DEVANAGARI_ONLY = re.compile(r"^[ऀ-ॿ ]+$")


def clean(s: str) -> str:
    """'Raw' scoring: only unicode NFC + whitespace collapse."""
    return " ".join(unicodedata.normalize("NFC", s or "").split())


def normalize(s: str) -> str:
    """'Normalised' scoring: also drops punctuation and zero-width joiners, lowercases stray Latin."""
    s = _ZERO_WIDTH.sub("", clean(s))
    return " ".join(_PUNCT.sub(" ", s).lower().split())


def wer_cer(refs, hyps, norm=False):
    f = normalize if norm else clean
    refs, hyps = [f(r) for r in refs], [f(h) for h in hyps]
    return {"wer": round(100 * jiwer.wer(refs, hyps), 2), "cer": round(100 * jiwer.cer(refs, hyps), 2), "n": len(refs)}


if __name__ == "__main__":
    assert normalize("नमस्कार, कसे आहात?  ") == "नमस्कार कसे आहात"
    assert normalize("प्रिंटस्‌ होते।") == "प्रिंटस् होते"
    assert wer_cer(["a b"], ["a b."], norm=True)["wer"] == 0 and wer_cer(["a b"], ["a b."])["wer"] == 50
    print("ok")
