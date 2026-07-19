"""Score a checkpoint on the full WMT14 newstest set with greedy and beam search.

Writes metrics.json and translations.jsonl (every hypothesis) to --out-dir.

    python -m src.eval --ckpt-file artifacts/ckpt_latest.pt --beam 5
"""

import argparse
import json
import os

import torch

from .data import stream_pairs, PAD_ID, BOS_ID, EOS_ID
from .model_fast import Seq2SeqFast
from .translate import get_tokenizers, translate, load_checkpoint


@torch.no_grad()
def beam_one(model, src_ids_rev, device, beam=5, max_len=80, alpha=0.6):
    """Beam search for one already-reversed source sentence."""
    src = torch.tensor(src_ids_rev, dtype=torch.long, device=device).unsqueeze(0)
    slen = torch.tensor([max(len(src_ids_rev), 1)])
    h, c = model.encoder(src, slen)
    h = h.repeat(1, beam, 1).contiguous()
    c = c.repeat(1, beam, 1).contiguous()
    seqs = torch.full((beam, 1), BOS_ID, dtype=torch.long, device=device)
    scores = torch.full((beam,), -1e9, device=device)
    scores[0] = 0.0                                     # one live beam to start
    fin_seqs, fin_scores = [], []

    for _ in range(max_len):
        out, (h, c) = model.decoder.lstm(model.decoder.embedding(seqs[:, -1:]), (h, c))
        logp = torch.log_softmax(model.decoder.output(out[:, -1]), dim=-1)
        cand = scores.unsqueeze(1) + logp
        V = cand.size(-1)
        top_v, top_i = cand.view(-1).topk(beam)
        b_idx = torch.div(top_i, V, rounding_mode="floor")
        tok = top_i % V
        seqs = torch.cat([seqs[b_idx], tok.unsqueeze(1)], dim=1)
        h = h[:, b_idx].contiguous()
        c = c[:, b_idx].contiguous()
        scores = top_v

        for i in (tok == EOS_ID).nonzero(as_tuple=True)[0].tolist():
            seq = seqs[i, 1:].tolist()
            if seq and seq[-1] == EOS_ID:
                seq = seq[:-1]
            lp = ((5 + len(seq)) / 6) ** alpha          # GNMT length penalty
            fin_seqs.append(seq)
            fin_scores.append(scores[i].item() / lp)
            scores[i] = -1e9
        if len(fin_seqs) >= beam and scores.max().item() < -1e8:
            break

    if not fin_seqs:
        return seqs[int(scores.argmax()), 1:].tolist()
    return fin_seqs[max(range(len(fin_seqs)), key=lambda i: fin_scores[i])]


def score(hyps, refs):
    import sacrebleu
    out = {
        "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
        "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
        "ter": round(sacrebleu.corpus_ter(hyps, [refs]).score, 2),
    }
    try:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        r = sum(sc.score(ref, hyp)["rougeL"].fmeasure for hyp, ref in zip(hyps, refs))
        out["rougeL"] = round(100 * r / len(hyps), 2)
    except ImportError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-artifact", default=None)
    ap.add_argument("--ckpt-file", default=None)
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--num-eval", type=int, default=None, help="cap sentences (default: full set)")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = load_checkpoint(args, device)
    src_tok, tgt_tok = get_tokenizers(args.out_dir)
    model = Seq2SeqFast(ck["cfg"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded step {ck['step']}", flush=True)

    val = stream_pairs("fr-en", "validation", limit=args.num_eval)
    srcs = [s for s, _ in val]
    refs = [t for _, t in val]
    print(f"newstest sentences: {len(srcs)}", flush=True)

    greedy = []
    for i in range(0, len(srcs), args.batch_size):
        greedy.extend(translate(model, src_tok, tgt_tok, srcs[i:i + args.batch_size], device, args.max_len))
        if i % (args.batch_size * 10) == 0:
            print(f"greedy {i}/{len(srcs)}", flush=True)

    beam = []
    for i, s in enumerate(srcs):
        ids = src_tok.encode(s).ids[: args.max_len][::-1]
        toks = [t for t in beam_one(model, ids, device, args.beam, args.max_len)
                if t not in (BOS_ID, EOS_ID, PAD_ID)]
        beam.append(tgt_tok.decode(toks))
        if i % 200 == 0:
            print(f"beam {i}/{len(srcs)}", flush=True)

    metrics = {"step": ck["step"], "n": len(srcs),
               "greedy": score(greedy, refs),
               f"beam{args.beam}": score(beam, refs)}
    print(json.dumps(metrics, indent=2), flush=True)

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out_dir, "translations.jsonl"), "w") as f:
        for s, r, g, b in zip(srcs, refs, greedy, beam):
            f.write(json.dumps({"en": s, "ref": r, "greedy": g, "beam": b}, ensure_ascii=False) + "\n")
    print("wrote metrics.json and translations.jsonl", flush=True)


if __name__ == "__main__":
    main()
