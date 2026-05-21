# Project handoff — RNN Translator

Context captured 2026-05-21 so this can be resumed on another machine.

## What this is

A small two-part project:

1. **Web UI** (`index.html`) — a minimalist translator page (English/Uzbek by
   default, 16 languages selectable) that talks to a **local Ollama** server.
2. **RNN seq2seq trainer** (`train.py`) — a PyTorch encoder-decoder with
   Bahdanau attention, trained on TSV pairs. The end goal is to train a custom
   translation model and wire it up in place of (or alongside) Ollama.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Browser UI. Fetches `GET /api/tags` for the model list and streams `POST /api/chat` for translations. Connection status pill + banner with setup instructions if Ollama is offline. |
| `train.py` | Training + inference CLI. BiGRU encoder, Bahdanau attention, GRU decoder, teacher forcing, greedy decode, checkpoint save/load. |
| `data/sample.tsv` | 96 English↔Uzbek pairs so the trainer is runnable out of the box. Replace with a larger corpus (e.g. Tatoeba `eng-uzb`) for real results. |
| `requirements.txt` | `torch>=2.1`. |
| `checkpoints/` | Created on first training run. Holds `best.pt` and `last.pt` (model weights + vocab + hparams in one file). |

## How to bring this up on a new machine

```powershell
# 1. clone (or copy this folder over)
git clone <your-remote> rnn-translator-app
cd rnn-translator-app

# 2. python deps
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Ollama for the web UI
#    - install from https://ollama.com
#    - pull at least one chat model
ollama pull llama3.2
#    - if opening index.html from disk (file://), start Ollama with
#      CORS open so the browser is allowed to call it:
$env:OLLAMA_ORIGINS="*"; ollama serve

# 4. open the page
start index.html
```

## How to train

```powershell
# smoke test — 2 epochs on the bundled tiny dataset
python train.py --epochs 2 --batch-size 16 --hid-dim 64 --emb-dim 64

# real run
python train.py --data data/sample.tsv --epochs 30

# inference from the saved checkpoint
python train.py --translate "hello, how are you?"
```

Verified working on Windows + Python 3 + torch 2.9.1+cu126 (CUDA). The sample
dataset is far too small to produce coherent translations — at 2 epochs it
collapses to `"men"` for every input. Plan on a real parallel corpus before
judging quality.

## Decisions / things worth knowing

- Web UI uses `POST /api/chat` (not `/api/generate`) for cleaner system-prompt
  handling and streaming.
- Strict system prompt: "Output ONLY the translation. No quotes, no
  commentary, no romanization, no source text, no explanations." — keeps the
  model from prefixing things like *Translation:*.
- Language pickers are click-to-open menus (16 languages); add more in the
  `LANGUAGES` array near the top of the script block in `index.html`.
- The RNN trainer stores **vocab + hparams + weights together** in one `.pt`
  file, so inference only needs `--checkpoint path/to/best.pt`.
- Tokenization is whitespace + punctuation split on lowercased NFKC text. Fine
  for a starter; swap in SentencePiece/BPE before scaling.
- Encoder is bidirectional GRU (1 layer by default), decoder is GRU (1 layer)
  with Bahdanau attention. Embedding + context + previous embedding all feed
  into the output projection.

## Open / next steps

- Train on a real corpus (Tatoeba `eng-uzb` is a reasonable starting point).
- Consider BPE/SentencePiece tokenization for OOV robustness.
- Once a checkpoint translates competently, expose `train.py` as a tiny HTTP
  server (Flask/FastAPI) and add it to the Model dropdown in `index.html`
  alongside Ollama models. The "Upload model" nav tab is already a placeholder
  for this flow.
- Tune `--teacher-forcing` (currently 0.6) and `--dropout` once data is real.

## Literal conversation transcript

Claude Code keeps the raw conversation as JSONL at:

```
%USERPROFILE%\.claude\projects\C--Users-Mardon-Programming-rnn-translator-app\<session-id>.jsonl
```

On this machine the current session file is
`ba154d4d-8e1c-49d5-a132-bd60343e582e.jsonl` (~670 KB). Copy that file to the
same relative path on the new machine and you can `claude --resume` to pick up
the exact thread. Note: project paths must match, so either keep the project
at `C:\Users\Mardon\Programming\rnn-translator-app` or rename the parent
directory to match (Claude Code encodes the path into the folder name with
slashes turned into dashes).

For most cases this `HANDOFF.md` plus the repo is enough — just `git push`
from here, `git pull` on the other device, point Claude Code at the project,
and tell it "read HANDOFF.md and continue."
