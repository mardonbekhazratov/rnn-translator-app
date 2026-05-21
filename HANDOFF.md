# Project handoff — RNN Translator

Context captured 2026-05-21 (refactored from a single-file UI to a Python-backed app on the same day).

## What this is

A small two-part project:

1. **Web app** — a minimalist translator page (English/Uzbek by default, 16
   languages selectable) served by a small **FastAPI backend** that proxies to a
   local **Ollama** server and enforces an allow-list of permitted models.
2. **RNN seq2seq trainer** (`train.py`) — a PyTorch encoder-decoder with
   Bahdanau attention, trained on TSV pairs. The end goal is to train a custom
   translation model and wire it up in place of (or alongside) Ollama.

## Files

| File | Purpose |
| --- | --- |
| `server.py` | FastAPI app. Serves `static/index.html`, exposes `/api/config`, `/api/models`, `/api/chat`. `/api/chat` is a streaming proxy onto Ollama's `POST /api/chat`. |
| `config.json` | The single place to edit allowed models, the language list, and Ollama URL. Re-read on every request — no restart needed. |
| `static/index.html` | Markup only. Links to `/static/css/styles.css` and `/static/js/app.js`. |
| `static/css/styles.css` | All styles. |
| `static/js/app.js` | All client logic: config + model fetch, debounced auto-translate, streaming response handling, language picker, swap/copy/clear. |
| `train.py` | Training + inference CLI. BiGRU encoder, Bahdanau attention, GRU decoder, teacher forcing, greedy decode, checkpoint save/load. |
| `data/sample.tsv` | 96 English↔Uzbek pairs so the trainer is runnable out of the box. Replace with a larger corpus (e.g. Tatoeba `eng-uzb`) for real results. |
| `requirements.txt` | `torch`, `fastapi`, `uvicorn`, `httpx`. |
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

# 3. Ollama (the actual translation engine)
#    - install from https://ollama.com
#    - pull at least one chat model
ollama pull llama3.2
ollama serve
#    Note: no more OLLAMA_ORIGINS=* workaround — the browser talks to the
#    Python server (same origin), not Ollama directly.

# 4. start the app
python server.py
#    → open http://127.0.0.1:8000
```

## Configuring allowed models

Edit `config.json`:

```json
{
  "ollama_url": "http://localhost:11434",
  "allowed_models": ["llama3.2", "qwen2.5"],
  "languages": ["English", "Uzbek", ...],
  "default_source_language": "English",
  "default_target_language": "Uzbek"
}
```

- `allowed_models: []` → no restriction (every installed Ollama model is shown).
- Names match either the exact tag (`llama3.2:latest`) or the family before `:`
  (`llama3.2` matches every tag in that family).
- The backend re-reads `config.json` on every request, so edits take effect on
  the next reload of the page — no server restart.
- The allow-list is enforced on both `/api/models` (filters the dropdown) and
  `/api/chat` (rejects requests for any non-allowed model with 403).

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

## Architecture notes

- **Why a backend at all** — the user asked: "what if I want to change the list
  of allowed models?" Stuffing that into JS would mean editing the bundle for
  every change. Pulling it into a backend also fixes the old CORS workaround
  (`OLLAMA_ORIGINS=*` is gone) and gives us a place to plug in the trained RNN
  later.
- `POST /api/chat` is a transparent streaming proxy — the body the frontend
  sends is forwarded verbatim to Ollama, and bytes come back through
  `aiter_raw()` unbuffered so token streaming still feels live.
- Strict system prompt is in `static/js/app.js` (`translate()`): "Output ONLY
  the translation. No quotes, no commentary, no romanization, no source text,
  no explanations." — keeps the model from prefixing things like *Translation:*.
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
- Once a checkpoint translates competently, expose it through `server.py` as a
  synthetic "model" in the dropdown (e.g. `rnn:best`) — the routing logic in
  `/api/chat` is already a natural place to branch on model name. The "Upload
  model" nav tab is already a placeholder for this flow.
- Tune `--teacher-forcing` (currently 0.6) and `--dropout` once data is real.

## Literal conversation transcript

Claude Code keeps the raw conversation as JSONL at:

```
~/.claude/projects/<project-folder-with-slashes-as-dashes>/<session-id>.jsonl
```

On this machine the session files live under
`~/.claude/projects/-Users-mardonhazratov-Documents-programming-rnn-translator-app/`.
Copy the relevant `.jsonl` to the same relative path on the new machine and you
can `claude --resume` to pick up the exact thread. Project paths must match, so
either keep the project at the same absolute path or rename the parent
directory accordingly (Claude Code encodes the path into the folder name with
slashes turned into dashes).

For most cases this `HANDOFF.md` plus the repo is enough — just `git push`
from here, `git pull` on the other device, point Claude Code at the project,
and tell it "read HANDOFF.md and continue."
