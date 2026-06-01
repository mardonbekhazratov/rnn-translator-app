"""Headless training script — same model as `train.ipynb` but suitable for
long unattended runs.

Recommended invocation:

    # In tmux/screen:
    tmux new -s train
    python train.py 2>&1 | tee train.log
    # Detach with Ctrl-b d; reattach later with `tmux attach -t train`.

    # Or with nohup:
    nohup python -u train.py > train.log 2>&1 &
    disown
    tail -f train.log

Both write checkpoints to `checkpoints/en2uz.pt` and `checkpoints/uz2en.pt`.
"""
from __future__ import annotations

import io
import math
import os
import random
import re
import tempfile
import time
import unicodedata
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Config -----------------------------------------------------------------

HIDDEN_SIZE = 256
BATCH_SIZE = 128
N_EPOCHS = 15
VOCAB_SIZE = 8000
MAX_LENGTH = 50
MAX_WORDS = 60
MAX_PAIRS = None  # None = use the whole OPUS-100 en-uz corpus

CKPT_DIR = Path(__file__).parent / "checkpoints"

# ---- Tokens / tokenizer -----------------------------------------------------

PAD_token, SOS_token, EOS_token, UNK_token = 0, 1, 2, 3


class SPTokenizer:
    def __init__(self, name, sp_model_bytes):
        self.name = name
        self.sp_model_bytes = sp_model_bytes
        self.sp = spm.SentencePieceProcessor()
        self.sp.LoadFromSerializedProto(sp_model_bytes)
        self.n_words = self.sp.GetPieceSize()

    def encode(self, text):
        return self.sp.EncodeAsIds(text)


def train_sentencepiece(texts, vocab_size, model_type="bpe",
                        input_sentence_size=200_000):
    """File-based SP training — Python iterator path is dramatically slower."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " "))
            f.write("\n")
        corpus_path = f.name
    try:
        buf = io.BytesIO()
        spm.SentencePieceTrainer.Train(
            input=corpus_path,
            model_writer=buf,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=1.0,
            pad_id=PAD_token, bos_id=SOS_token, eos_id=EOS_token, unk_id=UNK_token,
            pad_piece="<pad>", bos_piece="<s>", eos_piece="</s>", unk_piece="<unk>",
            num_threads=4,
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=True,
        )
        return buf.getvalue()
    finally:
        os.unlink(corpus_path)


# ---- Text normalization (must match rnn_infer.normalize_string) -------------

# Modern Uzbek Cyrillic -> Latin char map. Simple char-by-char substitution;
# OPUS-100 mixes both scripts so this collapses them to a single vocabulary.
_UZ_CYRL_MAP = {
    "а": "a",  "б": "b",  "в": "v",  "г": "g",  "д": "d",
    "е": "e",  "ё": "yo", "ж": "j",  "з": "z",  "и": "i",
    "й": "y",  "к": "k",  "л": "l",  "м": "m",  "н": "n",
    "о": "o",  "п": "p",  "р": "r",  "с": "s",  "т": "t",
    "у": "u",  "ф": "f",  "х": "x",  "ц": "s",  "ч": "ch",
    "ш": "sh", "ъ": "'",  "ы": "i",  "ь": "",
    "э": "e",  "ю": "yu", "я": "ya",
    "ў": "o'", "қ": "q",  "ғ": "g'", "ҳ": "h",
}


def _uz_cyrl_to_latin(s):
    return "".join(_UZ_CYRL_MAP.get(c, c) for c in s)


def normalizeString(s):
    s = unicodedata.normalize("NFC", s.lower().strip())
    for a in ("ʻ", "ʼ", "‘", "’", "`"):
        s = s.replace(a, "'")
    s = _uz_cyrl_to_latin(s)
    s = re.sub(r"([.!?,;:])", r" \1 ", s)
    s = re.sub(r"[^\w'.!?,;:]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ---- Data loading -----------------------------------------------------------

def load_opus_pairs(split="train", max_pairs=None):
    print(f"Loading Helsinki-NLP/opus-100 en-uz [{split}] ...", flush=True)
    ds = load_dataset("Helsinki-NLP/opus-100", "en-uz", split=split)
    pairs = []
    for ex in ds:
        t = ex["translation"]
        en = normalizeString(t["en"])
        uz = normalizeString(t["uz"])
        if en and uz:
            pairs.append((en, uz))
        if max_pairs and len(pairs) >= max_pairs:
            break
    print(f"Kept {len(pairs)} non-empty pairs", flush=True)
    return pairs


def filterPair(p):
    return len(p[0].split(" ")) < MAX_WORDS and len(p[1].split(" ")) < MAX_WORDS


def prepareData(direction, base_pairs, vocab_size=VOCAB_SIZE):
    if direction == "en-uz":
        src_name, tgt_name = "en", "uz"
        pairs = list(base_pairs)
    elif direction == "uz-en":
        src_name, tgt_name = "uz", "en"
        pairs = [(uz, en) for (en, uz) in base_pairs]
    else:
        raise ValueError(direction)

    pairs = [p for p in pairs if filterPair(p)]
    print(f"[{direction}] {len(pairs)} pairs after word-count filter", flush=True)

    print(f"[{direction}] training SP on {src_name} ...", flush=True)
    src_bytes = train_sentencepiece((p[0] for p in pairs), vocab_size=vocab_size)
    print(f"[{direction}] training SP on {tgt_name} ...", flush=True)
    tgt_bytes = train_sentencepiece((p[1] for p in pairs), vocab_size=vocab_size)

    input_lang = SPTokenizer(src_name, src_bytes)
    output_lang = SPTokenizer(tgt_name, tgt_bytes)
    print(f"[{direction}] vocab: {src_name}={input_lang.n_words} "
          f"{tgt_name}={output_lang.n_words}", flush=True)
    return input_lang, output_lang, pairs


def get_dataloader(batch_size, input_lang, output_lang, pairs):
    kept_src, kept_tgt = [], []
    over = 0
    for src, tgt in pairs:
        si = input_lang.encode(src) + [EOS_token]
        ti = output_lang.encode(tgt) + [EOS_token]
        if len(si) > MAX_LENGTH or len(ti) > MAX_LENGTH:
            over += 1
            continue
        kept_src.append(si)
        kept_tgt.append(ti)
    print(f"  kept {len(kept_src)} pairs (dropped {over} > MAX_LENGTH={MAX_LENGTH})",
          flush=True)

    n = len(kept_src)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int64)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int64)
    for i, (si, ti) in enumerate(zip(kept_src, kept_tgt)):
        input_ids[i, :len(si)] = si
        target_ids[i, :len(ti)] = ti

    data = TensorDataset(torch.LongTensor(input_ids).to(device),
                         torch.LongTensor(target_ids).to(device))
    return DataLoader(data, sampler=RandomSampler(data), batch_size=batch_size)


# ---- Model (must match rnn_infer.py) ----------------------------------------

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        return self.gru(embedded)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None,
                teacher_forcing_ratio=0.5):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.full((batch_size, 1), SOS_token,
                                   dtype=torch.long, device=device)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        for i in range(MAX_LENGTH):
            output, decoder_hidden, _ = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(output)
            if target_tensor is not None and random.random() < teacher_forcing_ratio:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = output.topk(1)
                decoder_input = topi.squeeze(-1).detach()
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return F.log_softmax(decoder_outputs, dim=-1)

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        output, hidden = self.gru(torch.cat((embedded, context), dim=2), hidden)
        return self.out(output), hidden, attn_weights


# ---- Training loop ----------------------------------------------------------

def train_epoch(dataloader, encoder, decoder, enc_opt, dec_opt, criterion,
                clip=1.0):
    total_loss = 0.0
    for input_tensor, target_tensor in dataloader:
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs = decoder(encoder_outputs, encoder_hidden, target_tensor)
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        enc_opt.step()
        dec_opt.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def asMinutes(s):
    m = math.floor(s / 60)
    return f"{m}m {int(s - m * 60)}s"


def train_model(dataloader, encoder, decoder, n_epochs, lr=1e-3):
    enc_opt = optim.Adam(encoder.parameters(), lr=lr)
    dec_opt = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.NLLLoss(ignore_index=PAD_token)
    start = time.time()
    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(dataloader, encoder, decoder, enc_opt, dec_opt,
                           criterion)
        elapsed = time.time() - start
        eta = elapsed / epoch * (n_epochs - epoch)
        print(f"  epoch {epoch:>2}/{n_epochs}  loss={loss:.4f}  "
              f"elapsed={asMinutes(elapsed)}  eta={asMinutes(eta)}", flush=True)


def save_model(direction, encoder, decoder, input_lang, output_lang):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f"{direction.replace('-', '2')}.pt"
    torch.save({
        "direction": direction,
        "hidden_size": HIDDEN_SIZE,
        "max_length": MAX_LENGTH,
        "encoder_state": encoder.state_dict(),
        "decoder_state": decoder.state_dict(),
        "input_sp_model": input_lang.sp_model_bytes,
        "output_sp_model": output_lang.sp_model_bytes,
        "input_lang_name": input_lang.name,
        "output_lang_name": output_lang.name,
    }, path)
    print(f"saved {path}", flush=True)


def build_and_train(direction, base_pairs):
    input_lang, output_lang, pairs = prepareData(direction, base_pairs)
    dataloader = get_dataloader(BATCH_SIZE, input_lang, output_lang, pairs)
    encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE).to(device)
    decoder = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words).to(device)
    print(f"[{direction}] training {N_EPOCHS} epochs on {device} ...", flush=True)
    train_model(dataloader, encoder, decoder, N_EPOCHS)
    save_model(direction, encoder, decoder, input_lang, output_lang)


def main():
    print(f"device={device}  cuda_available={torch.cuda.is_available()}", flush=True)
    base_pairs = load_opus_pairs(split="train", max_pairs=MAX_PAIRS)
    build_and_train("en-uz", base_pairs)
    build_and_train("uz-en", base_pairs)
    print("all done.", flush=True)


if __name__ == "__main__":
    main()
