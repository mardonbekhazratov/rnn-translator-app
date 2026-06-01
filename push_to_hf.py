"""Push the trained RNN translator checkpoints to the Hugging Face Hub.

Both directions live in a single repo so the model card and usage example
cover the whole translator.

Setup (one-time):
    pip install huggingface_hub
    huggingface-cli login        # or: export HF_TOKEN=hf_...

Run (after training has saved checkpoints/*.pt):
    python push_to_hf.py
    python push_to_hf.py --repo-id mardonbekhazratov/some-other-name
    python push_to_hf.py --private
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).parent
CKPT_DIR = ROOT / "checkpoints"
DEFAULT_REPO_ID = "mardonbekhazratov/rnn-en-uz-translator"

CHECKPOINTS = {
    "en2uz.pt": "English → Uzbek",
    "uz2en.pt": "Uzbek → English",
}

MODEL_CARD = """---
license: mit
language:
- en
- uz
library_name: pytorch
tags:
- translation
- seq2seq
- gru
- attention
- sentencepiece
- opus-100
pipeline_tag: translation
---

# English ↔ Uzbek RNN Translator

GRU encoder + Bahdanau (additive) attention + GRU decoder, trained on
[Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
`en-uz` with SentencePiece (BPE) subword tokenization. Two checkpoints, one
per direction:

| File         | Direction           |
|--------------|---------------------|
| `en2uz.pt`   | English → Uzbek     |
| `uz2en.pt`   | Uzbek → English     |

Each checkpoint is self-contained: it bundles the encoder/decoder weights
and the SentencePiece tokenizer model bytes for both source and target sides.

## Usage

The model is a custom PyTorch architecture (not a `transformers` model). Use
the inference module from
[the project repo](https://github.com/mardonbekhazratov/rnn-translator-app):

```python
from huggingface_hub import hf_hub_download
import rnn_infer  # from the project repo

ckpt_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="en2uz.pt",
)
# Point rnn_infer at the downloaded file (or copy it into ./checkpoints/).
translator = rnn_infer.Translator(ckpt_path)
print(translator.translate("hello, how are you?"))
```

## Checkpoint contents

Each `*.pt` file is a `torch.save` dict with:

- `encoder_state`, `decoder_state` — PyTorch `state_dict`s
- `input_sp_model`, `output_sp_model` — raw SentencePiece model bytes
- `hidden_size`, `max_length`, `direction`, plus language name metadata

## Training

See `train.ipynb` in the project repo. Trained on the full ~260k OPUS-100
`en-uz` parallel corpus, hidden size 256, SentencePiece vocab 8000 per side,
gradient clipping at 1.0, and scheduled sampling (teacher_forcing_ratio=0.5).
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                   help=f"Hub repo to push to (default: {DEFAULT_REPO_ID})")
    p.add_argument("--private", action="store_true",
                   help="Create the repo as private if it doesn't exist yet.")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HF token. Falls back to HF_TOKEN env var or cached login.")
    p.add_argument("--commit-message", default="Upload RNN translator checkpoints")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    missing = [name for name in CHECKPOINTS if not (CKPT_DIR / name).exists()]
    if missing:
        print(f"error: missing checkpoint(s) in {CKPT_DIR}: {missing}", file=sys.stderr)
        print("       run train.ipynb to completion first.", file=sys.stderr)
        return 1

    api = HfApi(token=args.token)
    api.create_repo(args.repo_id, repo_type="model",
                    private=args.private, exist_ok=True)
    print(f"repo ready: https://huggingface.co/{args.repo_id}")

    # Model card first so the page renders nicely while files upload.
    api.upload_file(
        path_or_fileobj=MODEL_CARD.format(repo_id=args.repo_id).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add model card",
    )
    print("uploaded README.md")

    for name, label in CHECKPOINTS.items():
        path = CKPT_DIR / name
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"uploading {name} ({label}, {size_mb:.1f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )

    print(f"\ndone — view at https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
