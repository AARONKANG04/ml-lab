"""WMT14 en-fr data: BPE tokenizers, a packed on-disk corpus, and batching.

Batches are (src, tgt_in, tgt_out). The source is reversed before the encoder
(Sutskever et al.), and the target is shifted so the decoder is trained on
tgt_in = [BOS, w1, ...] to predict tgt_out = [w1, ..., EOS].
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# PAD must be 0 to match nn.Embedding(padding_idx=0).
PAD_TOKEN, PAD_ID = "<pad>", 0
BOS_TOKEN, BOS_ID = "<bos>", 1
EOS_TOKEN, EOS_ID = "<eos>", 2
UNK_TOKEN, UNK_ID = "<unk>", 3
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

DEFAULT_DATASET = "wmt/wmt14"   # the legacy "wmt14" script loader breaks on datasets>=3
DEFAULT_CONFIG = "fr-en"
SRC_LANG = "en"
TGT_LANG = "fr"


def train_tokenizer(texts: Iterable[str], vocab_size: int, save_path: str | None = None):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS, show_progress=True)
    tokenizer.train_from_iterator(texts, trainer=trainer)

    for token, tid in zip(SPECIAL_TOKENS, (PAD_ID, BOS_ID, EOS_ID, UNK_ID)):
        assert tokenizer.token_to_id(token) == tid

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        tokenizer.save(save_path)
    return tokenizer


def load_tokenizer(path: str):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def stream_pairs(config: str = DEFAULT_CONFIG, split: str = "train", limit: int | None = None):
    """Stream up to `limit` (src, tgt) text pairs from WMT14."""
    from datasets import load_dataset

    ds = load_dataset(DEFAULT_DATASET, config, split=split, streaming=True)
    pairs = []
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        t = ex["translation"]
        pairs.append((t[SRC_LANG], t[TGT_LANG]))
    return pairs


def build_tokenizers(pairs, src_vocab_size, tgt_vocab_size, out_dir, train_lines=3_000_000):
    """Train src/tgt tokenizers on the first `train_lines` pairs, or reuse cached ones."""
    from itertools import islice

    src_path = os.path.join(out_dir, f"tokenizer_{SRC_LANG}.json")
    tgt_path = os.path.join(out_dir, f"tokenizer_{TGT_LANG}.json")
    if os.path.exists(src_path) and os.path.exists(tgt_path):
        return load_tokenizer(src_path), load_tokenizer(tgt_path)

    n = train_lines if train_lines is not None else len(pairs)
    src_tok = train_tokenizer((s for s, _ in islice(pairs, n)), src_vocab_size, src_path)
    tgt_tok = train_tokenizer((t for _, t in islice(pairs, n)), tgt_vocab_size, tgt_path)
    return src_tok, tgt_tok


def _pack_encoded(encodings, ids_chunks, len_chunks):
    flat, lens = [], []
    for e in encodings:
        flat.extend(e.ids)
        lens.append(len(e.ids))
    ids_chunks.append(np.asarray(flat, dtype=np.int32))
    len_chunks.append(np.asarray(lens, dtype=np.int64))


def load_or_build_packed(out_dir, split, src_tok, tgt_tok, max_samples=None, chunk=200_000):
    """Encode a split into flat int32 id arrays + offsets, cached to .npy.

    Storing ids as two contiguous arrays (one per language) with an offset array,
    instead of millions of Python lists, keeps memory to a few GB and lets a
    restart memory-map the cache instead of re-encoding. Raw text is dropped
    chunk by chunk while streaming.
    """
    prefix = os.path.join(out_dir, f"packed_{split}")
    files = {k: f"{prefix}_{k}.npy" for k in ("src_ids", "src_off", "tgt_ids", "tgt_off")}
    if all(os.path.exists(p) for p in files.values()):
        return {k: np.load(p, mmap_mode="r") for k, p in files.items()}

    from datasets import load_dataset
    ds = load_dataset(DEFAULT_DATASET, DEFAULT_CONFIG, split=split, streaming=True)

    src_ids, src_lens, tgt_ids, tgt_lens = [], [], [], []
    src_buf, tgt_buf = [], []

    def flush():
        _pack_encoded(src_tok.encode_batch(src_buf), src_ids, src_lens)
        _pack_encoded(tgt_tok.encode_batch(tgt_buf), tgt_ids, tgt_lens)
        src_buf.clear(); tgt_buf.clear()

    for i, ex in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        t = ex["translation"]
        src_buf.append(t[SRC_LANG]); tgt_buf.append(t[TGT_LANG])
        if len(src_buf) >= chunk:
            flush()
    if src_buf:
        flush()

    def finalize(ids_chunks, len_chunks):
        flat = np.concatenate(ids_chunks) if ids_chunks else np.zeros(0, dtype=np.int32)
        lens = np.concatenate(len_chunks) if len_chunks else np.zeros(0, dtype=np.int64)
        off = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=off[1:])
        return flat, off

    arrs = {}
    arrs["src_ids"], arrs["src_off"] = finalize(src_ids, src_lens)
    arrs["tgt_ids"], arrs["tgt_off"] = finalize(tgt_ids, tgt_lens)
    os.makedirs(out_dir, exist_ok=True)
    for k, p in files.items():
        np.save(p, arrs[k])
    return {k: np.load(p, mmap_mode="r") for k, p in files.items()}


class WMT14Dataset(Dataset):
    def __init__(self, packed: dict, max_len: int = 100):
        self.src_ids = packed["src_ids"]
        self.src_off = packed["src_off"]
        self.tgt_ids = packed["tgt_ids"]
        self.tgt_off = packed["tgt_off"]
        self.max_len = max_len

    def __len__(self):
        return len(self.src_off) - 1

    def __getitem__(self, idx):
        s = self.src_ids[self.src_off[idx]:self.src_off[idx + 1]][: self.max_len]
        t = self.tgt_ids[self.tgt_off[idx]:self.tgt_off[idx + 1]][: self.max_len]
        return s.tolist(), t.tolist()


@dataclass
class Collator:
    """Reverse the source, add BOS/EOS to the target, and pad to the batch max."""

    reverse_source: bool = True

    def __call__(self, batch):
        src_seqs, tgt_in_seqs, tgt_out_seqs, src_lens = [], [], [], []
        for src_ids, tgt_ids in batch:
            if self.reverse_source:
                src_ids = src_ids[::-1]
            src_seqs.append(torch.tensor(src_ids, dtype=torch.long))
            src_lens.append(len(src_ids))
            tgt_in_seqs.append(torch.tensor([BOS_ID] + tgt_ids, dtype=torch.long))
            tgt_out_seqs.append(torch.tensor(tgt_ids + [EOS_ID], dtype=torch.long))

        def pad(seqs):
            return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=PAD_ID)

        return {
            "src": pad(src_seqs),
            "tgt_in": pad(tgt_in_seqs),
            "tgt_out": pad(tgt_out_seqs),
            "src_len": torch.tensor(src_lens, dtype=torch.long),
        }


def build_dataloaders(out_dir="artifacts", batch_size=128, src_vocab_size=32_000,
                      tgt_vocab_size=32_000, max_len=100, max_train_samples=None,
                      reverse_source=True, num_workers=2):
    """Return ({'train', 'validation'} dataloaders, (src_tok, tgt_tok))."""
    src_path = os.path.join(out_dir, f"tokenizer_{SRC_LANG}.json")
    tgt_path = os.path.join(out_dir, f"tokenizer_{TGT_LANG}.json")
    if os.path.exists(src_path) and os.path.exists(tgt_path):
        src_tok, tgt_tok = load_tokenizer(src_path), load_tokenizer(tgt_path)
    else:
        tok_pairs = stream_pairs("fr-en", split="train", limit=3_000_000)
        src_tok, tgt_tok = build_tokenizers(tok_pairs, src_vocab_size, tgt_vocab_size, out_dir)
        del tok_pairs

    train_packed = load_or_build_packed(out_dir, "train", src_tok, tgt_tok, max_samples=max_train_samples)
    val_packed = load_or_build_packed(out_dir, "validation", src_tok, tgt_tok, max_samples=None)
    collate = Collator(reverse_source=reverse_source)

    loaders = {}
    for split, packed in (("train", train_packed), ("validation", val_packed)):
        ds = WMT14Dataset(packed, max_len=max_len)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
            num_workers=num_workers,
            drop_last=(split == "train"),
        )
    return loaders, (src_tok, tgt_tok)


if __name__ == "__main__":
    loaders, (src_tok, tgt_tok) = build_dataloaders(
        out_dir="artifacts",
        batch_size=4,
        src_vocab_size=8_000,
        tgt_vocab_size=8_000,
        max_train_samples=2_000,
        num_workers=0,
    )
    print(f"src vocab={src_tok.get_vocab_size()}  tgt vocab={tgt_tok.get_vocab_size()}")
    batch = next(iter(loaders["train"]))
    for k, v in batch.items():
        print(f"{k:8s} {tuple(v.shape)}")
    print("tgt_in[0] :", batch["tgt_in"][0].tolist())
    print("tgt_out[0]:", batch["tgt_out"][0].tolist())
