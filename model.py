"""Load Indic-Transcribe-core as a NeMo model (the HF `trust_remote_code` class is inference-only)."""
import os
import sys

from huggingface_hub import snapshot_download

REPO = "bodhan-ai/indic-transcribe-core"


def load(nemo_path=None):
    d = snapshot_download(REPO, allow_patterns=["nemo/*"])
    sys.path.insert(0, os.path.join(d, "nemo"))
    # The .nemo config names a tokenizer class that stock NeMo lacks; the repo's helper registers it
    # under NeMo's module path before restore (a plain import is NOT enough).
    from load_nemo import register_tokenizer
    register_tokenizer()
    from nemo.collections.asr.models import EncDecMultiTaskModel

    m = EncDecMultiTaskModel.restore_from(nemo_path or os.path.join(d, "nemo/indic_transcribe_core.nemo"), map_location="cuda")
    return m.eval()


def transcribe(m, manifest, c):
    # Pass our manifest (it carries sampling_rate), not a list of paths: for a path list NeMo writes a temp
    # manifest without it and lhotse then opens every file just to read its header (very slow on a network volume).
    out = m.transcribe(manifest, batch_size=c["evaluation"]["batch_size"], source_lang=c["model"]["lang"],
                       target_lang=c["model"]["lang"], pnc=c["model"]["pnc"], verbose=False)
    return [(getattr(h, "text", h) or "").strip() for h in out]
