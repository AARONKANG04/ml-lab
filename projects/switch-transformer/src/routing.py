import argparse
import json
import os

import numpy as np
import torch
import tiktoken

from model import GPT

VOCAB_SIZE = 50257


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/switch8.pt")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--windows", type=int, default=400)
    ap.add_argument("--min-count", type=int, default=25)
    ap.add_argument("--out", default="runs/inspect.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    a = ckpt["args"]
    model = GPT(VOCAB_SIZE, a["d_model"], a["n_heads"], a["n_layers"], a["d_ff"],
                max_seq_len=a["seq_len"], n_experts=a["experts"],
                capacity_factor=a["capacity_factor"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    seq = a["seq_len"]
    E = a["experts"]
    moe = model.moe_layers()
    counts = {i: np.zeros((E, VOCAB_SIZE), dtype=np.int64) for i, _ in moe}

    n = min(args.windows, (len(val) - 1) // seq)
    with torch.no_grad():
        for w in range(n):
            toks = val[w * seq:(w + 1) * seq].astype(np.int64)
            x = torch.from_numpy(toks)[None].to(device)
            model(x)
            for i, m in moe:
                ch = m.last_choice[0].cpu().numpy()
                np.add.at(counts[i], (ch, toks), 1)

    enc = tiktoken.get_encoding("gpt2")
    layers_out = []
    for i, _ in moe:
        c = counts[i]
        tok_tot = c.sum(axis=0)
        load = (c.sum(axis=1) / max(c.sum(), 1)).tolist()
        experts_out = []
        for e in range(E):
            share = np.where(tok_tot >= args.min_count, c[e] / np.maximum(tok_tot, 1), 0.0)
            top = np.argsort(-share)[:20]
            experts_out.append([
                {"token": enc.decode([int(t)]), "share": round(float(share[t]), 3),
                 "count": int(c[e, t])}
                for t in top if share[t] > 0
            ])
        layers_out.append({"layer": i, "load": [round(v, 4) for v in load],
                           "experts": experts_out})

    sample_w = 2
    toks = val[sample_w * seq:(sample_w + 1) * seq].astype(np.int64)
    x = torch.from_numpy(toks)[None].to(device)
    with torch.no_grad():
        model(x)
    sample = {
        "tokens": [enc.decode([int(t)]) for t in toks[:320]],
        "choices": {str(i): m.last_choice[0, :320].cpu().tolist() for i, m in moe},
    }

    with open(args.out, "w") as f:
        json.dump({"name": ckpt.get("name"), "n_experts": E, "windows": n,
                   "layers": layers_out, "sample": sample}, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
