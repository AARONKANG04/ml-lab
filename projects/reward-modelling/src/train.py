import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

DATASET = "argilla/ultrafeedback-binarized-preferences-cleaned"


class RewardModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name, dtype=torch.bfloat16,
                                              attn_implementation="sdpa")
        hidden = self.base.get_input_embeddings().weight.shape[1]
        self.head = nn.Linear(hidden, 1, dtype=torch.bfloat16)
        nn.init.normal_(self.head.weight, std=1.0 / (hidden + 1))
        nn.init.zeros_(self.head.bias)
        self.base.config.use_cache = False
        self.base.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        h = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(1) - 1
        pooled = h[torch.arange(h.shape[0], device=h.device), last]
        return self.head(pooled).squeeze(-1).float()


def encode(tokenizer, messages, max_len):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids[-max_len:]


def pad_batch(seqs, pad_id, device):
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


def score_pairs(model, tokenizer, rows, max_len, device, batch_size=16):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            seqs = [encode(tokenizer, r["chosen"], max_len) for r in chunk] \
                + [encode(tokenizer, r["rejected"], max_len) for r in chunk]
            ids, mask = pad_batch(seqs, tokenizer.pad_token_id, device)
            r = model(ids, mask)
            n = len(chunk)
            for j, row in enumerate(chunk):
                out.append({
                    "rc": round(r[j].item(), 4),
                    "rj": round(r[n + j].item(), 4),
                    "len_c": len(seqs[j]),
                    "len_j": len(seqs[n + j]),
                    "rating_c": row["chosen-rating"],
                    "rating_j": row["rejected-rating"],
                })
    model.train()
    return out


def accuracy(records):
    return float(np.mean([rec["rc"] > rec["rj"] for rec in records]))


def pick_examples(rows, records, k=4):
    margins = [rec["rc"] - rec["rj"] for rec in records]
    order = np.argsort(margins)
    picks = list(order[:k // 2]) + list(order[-k:][::-1][:k - k // 2])
    out = []
    for i in picks:
        row, rec = rows[i], records[i]
        out.append({
            "prompt": row["prompt"][:400],
            "chosen": row["chosen"][-1]["content"][:400],
            "rejected": row["rejected"][-1]["content"][:400],
            "rc": rec["rc"], "rj": rec["rj"],
            "rating_c": rec["rating_c"], "rating_j": rec["rating_j"],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--test-size", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="assets")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    ds = load_dataset(DATASET, split="train").shuffle(seed=args.seed)
    test = [ds[i] for i in range(args.test_size)]
    n_train = len(ds) - args.test_size
    if args.limit:
        n_train = min(n_train, args.limit)
    train_rows = [ds[args.test_size + i] for i in range(n_train)]
    print(f"{len(train_rows)} train pairs, {len(test)} test pairs", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = RewardModel(args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                            betas=(0.9, 0.95))
    steps = args.epochs * math.ceil(len(train_rows) / (args.batch_size * args.accum))
    warmup = max(10, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(
        (s + 1) / warmup, 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))

    history = {"train": [], "eval": []}
    eval_subset = test[:1000]
    rng = np.random.default_rng(args.seed)
    step = 0
    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    micro = 0
    for epoch in range(args.epochs):
        order = rng.permutation(len(train_rows))
        for i in range(0, len(order), args.batch_size):
            batch = [train_rows[j] for j in order[i:i + args.batch_size]]
            seqs = [encode(tokenizer, r["chosen"], args.max_len) for r in batch] \
                + [encode(tokenizer, r["rejected"], args.max_len) for r in batch]
            ids, mask = pad_batch(seqs, tokenizer.pad_token_id, device)
            r = model(ids, mask)
            n = len(batch)
            loss = -F.logsigmoid(r[:n] - r[n:]).mean() / args.accum
            loss.backward()
            micro += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            if step % 20 == 0:
                history["train"].append({"step": step,
                                         "loss": round(loss.item() * args.accum, 5)})
            if step % args.eval_every == 0 and step > 0:
                records = score_pairs(model, tokenizer, eval_subset,
                                      args.max_len, device)
                acc = accuracy(records)
                history["eval"].append({"step": step, "acc": acc})
                sps = step / (time.time() - t0)
                print(f"step {step:5d}/{steps}  loss {loss.item():.4f}  "
                      f"eval acc {acc:.4f}  {sps:.2f} steps/s", flush=True)
                (out_dir / "results.json").write_text(json.dumps(
                    {"config": vars(args), "history": history}))
            step += 1

    records = score_pairs(model, tokenizer, test, args.max_len, device)
    final_acc = accuracy(records)
    examples = pick_examples(test, records, k=8)
    (out_dir / "results.json").write_text(json.dumps({
        "config": vars(args), "history": history,
        "final": {"accuracy": final_acc, "records": records,
                  "examples": examples},
    }))
    torch.save(model.state_dict(), out_dir / "reward_model.pt")
    print(f"final test accuracy {final_acc:.4f} on {len(records)} pairs", flush=True)


if __name__ == "__main__":
    main()
