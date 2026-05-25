# RNN Translator

A small translator web app with two interchangeable backends:

1. **Ollama** — a FastAPI server proxies translation requests to a local Ollama
   instance, enforcing an allow-list of permitted models.
2. **A custom RNN** — a from-scratch PyTorch seq2seq model (GRU encoder/decoder
   with Bahdanau attention) trained on English↔Uzbek data, served locally by the
   same FastAPI app as a model named `rnn`.

Both show up in the same model dropdown; pick one and translate.

## Layout

| File | Purpose |
| --- | --- |
| `server.py` | FastAPI app. Serves the frontend and exposes `/api/config`, `/api/models`, `/api/chat`. Proxies to Ollama, and serves the local `rnn` model directly. |
| `rnn_infer.py` | Loads the trained checkpoints and runs greedy decoding. Mirrors the notebook architecture so `state_dict`s load cleanly. |
| `train.ipynb` | Trains the two RNN models (English→Uzbek and Uzbek→English) on the OPUS-100 corpus and saves checkpoints. |
| `config.json` | Ollama URL, allowed-models list, language list, and UI defaults. Re-read on every request — no restart needed. |
| `static/index.html` | Markup. Links `static/css/styles.css` and `static/js/app.js`. |
| `static/js/app.js` | Client logic: config/model fetch, debounced auto-translate, streaming, language picker, swap/copy/clear. |
| `requirements.txt` | Runtime + training dependencies. |
| `checkpoints/` | Created by training. Holds `en2uz.pt` and `uz2en.pt` (weights + vocab + config). Git-ignored. |

## Setup

```bash
python -m venv venv
source venv/bin/activate            # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the app

```bash
python server.py                    # → http://127.0.0.1:8000
```

To use the **Ollama** backend, also run Ollama and pull a chat model:

```bash
ollama pull llama3.2
ollama serve
```

The `rnn` model needs no Ollama — it appears in the dropdown as soon as a
trained checkpoint exists, even if Ollama is offline.

## Train the RNN models

Open `train.ipynb` and run it top to bottom. It:

- downloads the [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100/viewer/en-uz)
  `en-uz` corpus (cached after the first run),
- trains **two separate models** — one per direction — using a GRU encoder/decoder
  with Bahdanau attention,
- saves `checkpoints/en2uz.pt` and `checkpoints/uz2en.pt`.

Knobs at the top of the training cell trade speed for quality:

- `MAX_PAIRS` — how much of the ~260k-pair corpus to use (`None` = all, much slower on CPU).
- `min_count` — drop rare words to shrink the vocab and speed up the output softmax.
- `hidden_size`, `batch_size`, `n_epochs` — the usual capacity/throughput dials.

This is a word-level RNN trained from scratch: it learns plausible structure with
enough data/epochs on a GPU, but won't match production translation quality. For
better results, scale the data, switch to subword (BPE/SentencePiece) tokenization,
and train on a GPU.

## How the `rnn` model is served

1. The frontend sends the selected source/target languages with each request.
2. `server.py` routes any request for model `rnn` to `rnn_infer.translate(...)`
   instead of Ollama, picking the checkpoint from those languages
   (English→Uzbek → `en2uz.pt`, Uzbek→English → `uz2en.pt`).
3. The translation is streamed back word-by-word in Ollama's NDJSON chat format,
   so the frontend renders it the same way as a proxied model.

Only English↔Uzbek is supported; any other language pair returns an error asking
you to train that direction first.

## Configuring allowed Ollama models

Edit `config.json`:

```json
{
  "ollama_url": "http://localhost:11434",
  "allowed_models": ["llama3.2", "qwen2.5"],
  "languages": ["English", "Uzbek", "..."],
  "default_source_language": "English",
  "default_target_language": "Uzbek"
}
```

- `allowed_models: []` → no restriction (every installed Ollama model is shown).
- Names match the exact tag (`llama3.2:latest`) or the family before `:` (`llama3.2`).
- The allow-list is enforced on both `/api/models` (filters the dropdown) and
  `/api/chat` (rejects non-allowed models with 403). The local `rnn` model bypasses it.
- Edits take effect on the next page reload — no server restart.
