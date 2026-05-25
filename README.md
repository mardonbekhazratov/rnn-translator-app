# RNN Translator

A small English↔Uzbek translator web app powered by a custom PyTorch seq2seq
model (GRU encoder/decoder with Bahdanau attention) trained from scratch. The
model runs in-process inside a FastAPI server — there is no external model
service to install or run.

## Layout

| File | Purpose |
| --- | --- |
| `server.py` | FastAPI app. Serves the frontend and exposes `/api/config`, `/api/models`, `/api/chat`. Runs the trained RNN in-process. |
| `rnn_infer.py` | Loads the trained checkpoints and runs greedy decoding. Mirrors the notebook architecture so `state_dict`s load cleanly. |
| `train.ipynb` | Trains the two RNN models (English→Uzbek and Uzbek→English) on the OPUS-100 corpus and saves checkpoints. |
| `config.json` | Language list and UI defaults. Re-read on every request — no restart needed. |
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

## Train the models

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

## Run the app

```bash
python server.py                    # → http://127.0.0.1:8000
```

The model named `rnn` appears in the dropdown as soon as a trained checkpoint
exists. Until you've trained at least one direction, the app reports that no
trained model was found.

### How translation works

1. The frontend sends the selected source/target languages with each request.
2. `server.py` runs `rnn_infer.translate(...)`, picking the checkpoint from those
   languages (English→Uzbek → `en2uz.pt`, Uzbek→English → `uz2en.pt`).
3. The translation is streamed back word-by-word as NDJSON so the UI renders it live.

Only English↔Uzbek is supported. To add more language pairs, train them in the
notebook and add the languages to `config.json`.
