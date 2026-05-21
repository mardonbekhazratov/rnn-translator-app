"""
RNN sequence-to-sequence trainer for machine translation.

Architecture:
  - Embedding -> bidirectional GRU encoder
  - Bahdanau additive attention
  - Embedding -> GRU decoder with attention context
  - Teacher forcing during training, greedy/argmax decode at inference

Data format:
  A TSV file with two columns per line: source<TAB>target
  See data/sample.tsv for an example.

Usage:
  python train.py --data data/sample.tsv --epochs 30
  python train.py --data data/sample.tsv --translate "hello, how are you?"
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ----------------------------- Vocabulary -----------------------------

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIALS = [PAD, SOS, EOS, UNK]


class Vocab:
    def __init__(self, tokens: list[str]):
        self.itos = list(SPECIALS) + [t for t in tokens if t not in SPECIALS]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad(self) -> int: return self.stoi[PAD]
    @property
    def sos(self) -> int: return self.stoi[SOS]
    @property
    def eos(self) -> int: return self.stoi[EOS]
    @property
    def unk(self) -> int: return self.stoi[UNK]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, self.unk) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        out = []
        for i in ids:
            tok = self.itos[i]
            if tok == EOS: break
            if tok in (PAD, SOS): continue
            out.append(tok)
        return out

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocab":
        v = cls.__new__(cls)
        v.itos = d["itos"]
        v.stoi = {t: i for i, t in enumerate(v.itos)}
        return v


# ----------------------------- Tokenization -----------------------------

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip().lower()
    s = re.sub(r"([.!?,;:])", r" \1 ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tokenize(s: str) -> list[str]:
    return normalize(s).split()


# ----------------------------- Dataset -----------------------------

def load_pairs(path: Path, max_len: int) -> list[tuple[list[str], list[str]]]:
    pairs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            src, tgt = line.split("\t", 1)
            s_tok, t_tok = tokenize(src), tokenize(tgt)
            if not s_tok or not t_tok:
                continue
            if len(s_tok) > max_len or len(t_tok) > max_len:
                continue
            pairs.append((s_tok, t_tok))
    return pairs


def build_vocab(pairs, side: int, min_freq: int = 1) -> Vocab:
    freq: dict[str, int] = {}
    for p in pairs:
        for tok in p[side]:
            freq[tok] = freq.get(tok, 0) + 1
    tokens = sorted([t for t, c in freq.items() if c >= min_freq])
    return Vocab(tokens)


class PairsDataset(Dataset):
    def __init__(self, pairs, src_vocab: Vocab, tgt_vocab: Vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        s, t = self.pairs[i]
        src_ids = self.src_vocab.encode(s) + [self.src_vocab.eos]
        tgt_ids = [self.tgt_vocab.sos] + self.tgt_vocab.encode(t) + [self.tgt_vocab.eos]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate(batch, src_pad: int, tgt_pad: int):
    batch.sort(key=lambda x: x[0].size(0), reverse=True)
    src, tgt = zip(*batch)
    src_lens = torch.tensor([s.size(0) for s in src], dtype=torch.long)
    tgt_lens = torch.tensor([t.size(0) for t in tgt], dtype=torch.long)
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=src_pad)
    tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=tgt_pad)
    return src, src_lens, tgt, tgt_lens


# ----------------------------- Model -----------------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout if n_layers > 1 else 0.0,
                         bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lens):
        # src: [B, S]
        emb = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=True)
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # hidden: [2*n_layers, B, H] -> combine the last layer's two directions
        last_fwd = hidden[-2]
        last_bwd = hidden[-1]
        h0 = torch.tanh(self.fc(torch.cat([last_fwd, last_bwd], dim=1)))  # [B, H]
        return outputs, h0  # outputs: [B, S, 2H]


class BahdanauAttention(nn.Module):
    def __init__(self, hid_dim: int):
        super().__init__()
        self.W_h = nn.Linear(hid_dim * 2, hid_dim, bias=False)
        self.W_s = nn.Linear(hid_dim, hid_dim, bias=False)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, s, enc_out, src_mask):
        # s: [B, H]; enc_out: [B, S, 2H]; src_mask: [B, S] (1=keep, 0=pad)
        s_exp = s.unsqueeze(1).expand(-1, enc_out.size(1), -1)  # [B, S, H]
        e = self.v(torch.tanh(self.W_h(enc_out) + self.W_s(s_exp))).squeeze(-1)  # [B, S]
        e = e.masked_fill(~src_mask, -1e9)
        a = F.softmax(e, dim=1)  # [B, S]
        ctx = torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)  # [B, 2H]
        return ctx, a


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.attn = BahdanauAttention(hid_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim * 2, hid_dim, num_layers=n_layers,
                         dropout=dropout if n_layers > 1 else 0.0, batch_first=True)
        self.fc_out = nn.Linear(hid_dim * 3 + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers

    def forward(self, y_prev, hidden, enc_out, src_mask):
        # y_prev: [B], hidden: [n_layers, B, H], enc_out: [B, S, 2H]
        emb = self.dropout(self.embedding(y_prev)).unsqueeze(1)  # [B, 1, E]
        s_top = hidden[-1]  # [B, H]
        ctx, attn = self.attn(s_top, enc_out, src_mask)  # ctx: [B, 2H]
        rnn_in = torch.cat([emb, ctx.unsqueeze(1)], dim=2)  # [B, 1, E+2H]
        out, hidden = self.rnn(rnn_in, hidden)             # out: [B, 1, H]
        out = out.squeeze(1)
        logits = self.fc_out(torch.cat([out, ctx, emb.squeeze(1)], dim=1))  # [B, V]
        return logits, hidden, attn


class Seq2Seq(nn.Module):
    def __init__(self, enc: Encoder, dec: Decoder, src_pad: int, tgt_pad: int):
        super().__init__()
        self.encoder = enc
        self.decoder = dec
        self.src_pad = src_pad
        self.tgt_pad = tgt_pad

    def _init_dec_hidden(self, h0: torch.Tensor) -> torch.Tensor:
        # broadcast encoder summary to all decoder layers
        return h0.unsqueeze(0).expand(self.decoder.n_layers, -1, -1).contiguous()

    def forward(self, src, src_lens, tgt, teacher_forcing: float = 0.5):
        B, T = tgt.size()
        V = self.decoder.fc_out.out_features
        src_mask = (src != self.src_pad)
        enc_out, h0 = self.encoder(src, src_lens)
        hidden = self._init_dec_hidden(h0)

        outputs = torch.zeros(B, T - 1, V, device=src.device)
        y_t = tgt[:, 0]  # <sos>
        for t in range(1, T):
            logits, hidden, _ = self.decoder(y_t, hidden, enc_out, src_mask)
            outputs[:, t - 1] = logits
            use_tf = torch.rand(1, device=src.device).item() < teacher_forcing
            y_t = tgt[:, t] if use_tf else logits.argmax(dim=1)
        return outputs

    @torch.no_grad()
    def greedy(self, src, src_lens, sos: int, eos: int, max_len: int = 60):
        self.eval()
        B = src.size(0)
        src_mask = (src != self.src_pad)
        enc_out, h0 = self.encoder(src, src_lens)
        hidden = self._init_dec_hidden(h0)
        y_t = torch.full((B,), sos, dtype=torch.long, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        out_ids = [[] for _ in range(B)]
        for _ in range(max_len):
            logits, hidden, _ = self.decoder(y_t, hidden, enc_out, src_mask)
            y_t = logits.argmax(dim=1)
            for i in range(B):
                if not finished[i]:
                    tok = y_t[i].item()
                    if tok == eos:
                        finished[i] = True
                    else:
                        out_ids[i].append(tok)
            if finished.all(): break
        return out_ids


# ----------------------------- Train loop -----------------------------

@dataclass
class Hparams:
    emb_dim: int = 192
    hid_dim: int = 256
    enc_layers: int = 1
    dec_layers: int = 1
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 30
    max_len: int = 50
    teacher_forcing: float = 0.6
    grad_clip: float = 1.0
    seed: int = 0
    val_split: float = 0.1


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(pairs, src_vocab, tgt_vocab, hp: Hparams):
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * hp.val_split)) if len(pairs) > 4 else 0
    train_pairs = pairs[n_val:] if n_val else pairs
    val_pairs = pairs[:n_val]

    train_ds = PairsDataset(train_pairs, src_vocab, tgt_vocab)
    val_ds = PairsDataset(val_pairs, src_vocab, tgt_vocab) if val_pairs else None

    def _collate(b): return collate(b, src_vocab.pad, tgt_vocab.pad)

    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=hp.batch_size, shuffle=False, collate_fn=_collate) if val_ds else None
    return train_loader, val_loader


def run_epoch(model, loader, optim, criterion, device, hp, train: bool):
    model.train(train)
    total_loss, total_tok = 0.0, 0
    for src, src_lens, tgt, _ in loader:
        src, tgt = src.to(device), tgt.to(device)
        if train:
            optim.zero_grad()
        logits = model(src, src_lens, tgt, teacher_forcing=hp.teacher_forcing if train else 0.0)
        # logits: [B, T-1, V], target: tgt[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            optim.step()
        n_tok = (tgt[:, 1:] != model.tgt_pad).sum().item()
        total_loss += loss.item() * n_tok
        total_tok += n_tok
    return total_loss / max(total_tok, 1)


def train(args):
    hp = Hparams(
        emb_dim=args.emb_dim, hid_dim=args.hid_dim,
        enc_layers=args.enc_layers, dec_layers=args.dec_layers,
        dropout=args.dropout, lr=args.lr, batch_size=args.batch_size,
        epochs=args.epochs, max_len=args.max_len,
        teacher_forcing=args.teacher_forcing, seed=args.seed,
    )
    set_seed(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = load_pairs(Path(args.data), hp.max_len)
    if not pairs:
        raise SystemExit(f"No usable pairs found in {args.data}")
    print(f"Loaded {len(pairs)} pairs")

    src_vocab = build_vocab(pairs, side=0)
    tgt_vocab = build_vocab(pairs, side=1)
    print(f"Vocab: src={len(src_vocab)}  tgt={len(tgt_vocab)}")

    train_loader, val_loader = make_loaders(pairs, src_vocab, tgt_vocab, hp)

    enc = Encoder(len(src_vocab), hp.emb_dim, hp.hid_dim, hp.enc_layers, hp.dropout, src_vocab.pad)
    dec = Decoder(len(tgt_vocab), hp.emb_dim, hp.hid_dim, hp.dec_layers, hp.dropout, tgt_vocab.pad)
    model = Seq2Seq(enc, dec, src_vocab.pad, tgt_vocab.pad).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params on {device}")

    optim = torch.optim.Adam(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = math.inf
    for ep in range(1, hp.epochs + 1):
        t0 = time.time()
        tr_loss = run_epoch(model, train_loader, optim, criterion, device, hp, train=True)
        val_loss = run_epoch(model, val_loader, optim, criterion, device, hp, train=False) if val_loader else float("nan")
        dt = time.time() - t0
        print(f"epoch {ep:3d}/{hp.epochs}  train {tr_loss:.4f}  "
              f"val {val_loss:.4f}  ppl {math.exp(min(20, tr_loss)):.2f}  ({dt:.1f}s)")

        score = val_loss if val_loader else tr_loss
        if score < best_val:
            best_val = score
            save_checkpoint(out_dir / "best.pt", model, src_vocab, tgt_vocab, hp)

    save_checkpoint(out_dir / "last.pt", model, src_vocab, tgt_vocab, hp)
    print(f"Saved checkpoints to {out_dir.resolve()}")

    # quick sanity demo
    demo = [pairs[i][0] for i in range(min(3, len(pairs)))]
    for toks in demo:
        out = translate_one(model, src_vocab, tgt_vocab, " ".join(toks), device)
        print(f"  src: {' '.join(toks)}")
        print(f"  tgt: {out}")


def save_checkpoint(path: Path, model: Seq2Seq, src_vocab: Vocab, tgt_vocab: Vocab, hp: Hparams):
    torch.save({
        "model_state": model.state_dict(),
        "src_vocab": src_vocab.to_dict(),
        "tgt_vocab": tgt_vocab.to_dict(),
        "hp": asdict(hp),
    }, path)


def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device)
    src_vocab = Vocab.from_dict(ckpt["src_vocab"])
    tgt_vocab = Vocab.from_dict(ckpt["tgt_vocab"])
    hp_dict = ckpt["hp"]
    hp = Hparams(**hp_dict)
    enc = Encoder(len(src_vocab), hp.emb_dim, hp.hid_dim, hp.enc_layers, hp.dropout, src_vocab.pad)
    dec = Decoder(len(tgt_vocab), hp.emb_dim, hp.hid_dim, hp.dec_layers, hp.dropout, tgt_vocab.pad)
    model = Seq2Seq(enc, dec, src_vocab.pad, tgt_vocab.pad).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, src_vocab, tgt_vocab, hp


@torch.no_grad()
def translate_one(model: Seq2Seq, src_vocab: Vocab, tgt_vocab: Vocab, text: str, device) -> str:
    toks = tokenize(text)
    if not toks:
        return ""
    ids = src_vocab.encode(toks) + [src_vocab.eos]
    src = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    src_lens = torch.tensor([len(ids)], dtype=torch.long)
    out_ids = model.greedy(src, src_lens, tgt_vocab.sos, tgt_vocab.eos, max_len=80)[0]
    return " ".join(tgt_vocab.decode(out_ids))


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint or Path(args.out) / "best.pt")
    model, src_vocab, tgt_vocab, _ = load_checkpoint(ckpt_path, device)
    print(translate_one(model, src_vocab, tgt_vocab, args.translate, device))


# ----------------------------- CLI -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sample.tsv")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--checkpoint", default=None, help="Path to .pt for --translate (defaults to {out}/best.pt)")
    p.add_argument("--translate", default=None, help="If set, load checkpoint and translate this text")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--emb-dim", type=int, default=192)
    p.add_argument("--hid-dim", type=int, default=256)
    p.add_argument("--enc-layers", type=int, default=1)
    p.add_argument("--dec-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--max-len", type=int, default=50)
    p.add_argument("--teacher-forcing", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    if args.translate is not None:
        infer(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
