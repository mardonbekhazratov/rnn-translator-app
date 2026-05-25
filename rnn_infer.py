"""Local inference for the trained RNN seq2seq translator.

Loads the checkpoints produced by `train.ipynb` (`checkpoints/en2uz.pt`,
`checkpoints/uz2en.pt`) and translates text with greedy decoding. The model
architecture here must match the notebook exactly so the saved `state_dict`s
load cleanly.

Used by `server.py` to serve a synthetic `rnn` model alongside Ollama.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent
CKPT_DIR = ROOT / "checkpoints"
DEVICE = torch.device("cpu")  # inference is light; CPU keeps deploy simple

# Special tokens — must match train.ipynb.
PAD_token, SOS_token, EOS_token, UNK_token = 0, 1, 2, 3

# UI language name -> short code used in checkpoint filenames.
LANG_CODES = {"english": "en", "uzbek": "uz"}


# ----------------------------- Text normalization -----------------------------
# Must match `normalizeString` in train.ipynb so inference sees training-shaped text.

def normalize_string(s: str) -> str:
    s = unicodedata.normalize("NFC", s.lower().strip())
    for a in ("ʻ", "ʼ", "‘", "’", "`"):
        s = s.replace(a, "'")
    s = re.sub(r"([.!?,;:])", r" \1 ", s)
    s = re.sub(r"[^\w'.!?,;:]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def detokenize(words: list[str]) -> str:
    """Re-join tokens into readable text (no space before punctuation)."""
    text = " ".join(words)
    text = re.sub(r"\s+([.!?,;:])", r"\1", text)
    return text.strip()


# ----------------------------- Model (mirrors the notebook) -----------------------------

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


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

    def forward(self, encoder_outputs, encoder_hidden, max_length):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=encoder_outputs.device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for _ in range(max_length):
            decoder_output, decoder_hidden, _ = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)
        return output, hidden, attn_weights


# ----------------------------- Loading + translation -----------------------------

class Translator:
    def __init__(self, ckpt_path: Path):
        # Our own trusted checkpoint; full unpickling is required for the vocab dicts.
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        self.direction = ckpt["direction"]
        self.max_length = ckpt["max_length"]
        hidden_size = ckpt["hidden_size"]
        self.in_lang = ckpt["input_lang"]
        self.out_lang = ckpt["output_lang"]

        self.encoder = EncoderRNN(self.in_lang["n_words"], hidden_size).to(DEVICE)
        self.decoder = AttnDecoderRNN(hidden_size, self.out_lang["n_words"]).to(DEVICE)
        self.encoder.load_state_dict(ckpt["encoder_state"])
        self.decoder.load_state_dict(ckpt["decoder_state"])
        self.encoder.eval()
        self.decoder.eval()

    @torch.no_grad()
    def translate(self, text: str) -> str:
        text = normalize_string(text)
        if not text:
            return ""
        w2i = self.in_lang["word2index"]
        ids = [w2i.get(w, UNK_token) for w in text.split(" ")]
        ids.append(EOS_token)
        input_tensor = torch.tensor(ids, dtype=torch.long, device=DEVICE).view(1, -1)

        encoder_outputs, encoder_hidden = self.encoder(input_tensor)
        decoder_outputs = self.decoder(encoder_outputs, encoder_hidden, self.max_length)

        _, topi = decoder_outputs.topk(1)
        i2w = self.out_lang["index2word"]
        words = []
        for idx in topi.squeeze(0).squeeze(-1):
            tok = idx.item()
            if tok == EOS_token:
                break
            if tok in (PAD_token, UNK_token, SOS_token):
                continue
            words.append(i2w.get(tok, ""))
        return detokenize([w for w in words if w])


def _ckpt_path(direction: str) -> Path:
    # direction "en-uz" -> file "en2uz.pt"
    return CKPT_DIR / f"{direction.replace('-', '2')}.pt"


def direction_for(source_language: str, target_language: str) -> str | None:
    src = LANG_CODES.get((source_language or "").strip().lower())
    tgt = LANG_CODES.get((target_language or "").strip().lower())
    if not src or not tgt or src == tgt:
        return None
    return f"{src}-{tgt}"


def available_directions() -> list[str]:
    out = []
    for direction in ("en-uz", "uz-en"):
        if _ckpt_path(direction).exists():
            out.append(direction)
    return out


@lru_cache(maxsize=4)
def _get_translator(direction: str) -> Translator:
    return Translator(_ckpt_path(direction))


def translate(source_language: str, target_language: str, text: str) -> str:
    """Raise ValueError if the language pair has no trained checkpoint."""
    direction = direction_for(source_language, target_language)
    if direction is None or direction not in available_directions():
        supported = ", ".join(available_directions()) or "none trained yet"
        raise ValueError(
            f"The RNN model has no checkpoint for {source_language!r} -> {target_language!r}. "
            f"Supported directions: {supported}. Train them in train.ipynb first."
        )
    return _get_translator(direction).translate(text)
