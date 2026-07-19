"""Translate en-fr with a trained checkpoint and print a few examples.

    python -m src.translate --ckpt-file artifacts/ckpt_latest.pt --num-eval 20
    python -m src.translate --ckpt-artifact entity/project/seq2seq-ckpt:latest
"""

import argparse
import os

import torch

from .data import (build_tokenizers, stream_pairs, load_tokenizer, PAD_ID,
                   BOS_ID, EOS_ID, SRC_LANG, TGT_LANG)
from .model_fast import Seq2SeqFast


def get_tokenizers(out_dir):
    sp = os.path.join(out_dir, f"tokenizer_{SRC_LANG}.json")
    tp = os.path.join(out_dir, f"tokenizer_{TGT_LANG}.json")
    if os.path.exists(sp) and os.path.exists(tp):
        return load_tokenizer(sp), load_tokenizer(tp)
    # no cache: retrain on the same first-3M stream, which is deterministic
    pairs = stream_pairs("fr-en", "train", limit=3_000_000)
    return build_tokenizers(pairs, 32_000, 32_000, out_dir)


@torch.no_grad()
def translate(model, src_tok, tgt_tok, sentences, device, max_len=80, reverse=True):
    model.eval()
    enc = [src_tok.encode(s).ids[:max_len] for s in sentences]
    if reverse:
        enc = [e[::-1] for e in enc]
    lens = torch.tensor([max(len(e), 1) for e in enc])
    width = max(lens).item()
    src = torch.full((len(enc), width), PAD_ID, dtype=torch.long)
    for i, e in enumerate(enc):
        if e:
            src[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    src = src.to(device)
    state = model.encoder(src, lens)
    outs = model.decoder.generate(state, BOS_ID, EOS_ID, max_len, device)
    results = []
    for ids in outs:
        ids = [i for i in ids if i not in (BOS_ID, EOS_ID, PAD_ID)]
        results.append(tgt_tok.decode(ids))
    return results


def load_checkpoint(args, device):
    ckpt_file = args.ckpt_file
    if ckpt_file is None:
        import wandb
        d = wandb.Api().artifact(args.ckpt_artifact).download()
        ckpt_file = os.path.join(d, next(f for f in os.listdir(d) if f.endswith(".pt")))
    return torch.load(ckpt_file, map_location=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-artifact", default=None, help="W&B artifact ref, e.g. entity/project/seq2seq-ckpt:latest")
    ap.add_argument("--ckpt-file", default=None, help="local checkpoint path (skips W&B download)")
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--num-eval", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = load_checkpoint(args, device)
    src_tok, tgt_tok = get_tokenizers(args.out_dir)
    assert src_tok.get_vocab_size() == ck["cfg"]["src_vocab_size"], "src tokenizer does not match checkpoint"
    assert tgt_tok.get_vocab_size() == ck["cfg"]["tgt_vocab_size"], "tgt tokenizer does not match checkpoint"

    model = Seq2SeqFast(ck["cfg"]).to(device)
    model.load_state_dict(ck["model"])
    print(f"loaded step {ck['step']} onto {device}")

    val = stream_pairs("fr-en", "validation", limit=args.num_eval)
    srcs = [s for s, _ in val]
    refs = [t for _, t in val]

    hyps = []
    for i in range(0, len(srcs), args.batch_size):
        hyps.extend(translate(model, src_tok, tgt_tok, srcs[i:i + args.batch_size], device))

    for i in range(min(args.show, len(srcs))):
        print(f"EN : {srcs[i]}")
        print(f"HYP: {hyps[i]}")
        print(f"REF: {refs[i]}\n")

    try:
        import sacrebleu
        print(f"BLEU (greedy, n={len(hyps)}): {sacrebleu.corpus_bleu(hyps, [refs]).score:.2f}")
    except ImportError:
        print("sacrebleu not installed; skipping BLEU")


if __name__ == "__main__":
    main()
