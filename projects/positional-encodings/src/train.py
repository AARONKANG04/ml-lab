import argparse
import json
import math
import os

import numpy as np
import torch

from model_fast import GPT

VOCAB_SIZE = 50257
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192]


def prepare(data_dir):
    import tiktoken
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    os.makedirs(data_dir, exist_ok=True)
    for split, hf_split in (("train", "train"), ("val", "validation")):
        path = os.path.join(data_dir, f"{split}.bin")
        if os.path.exists(path):
            continue
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=hf_split)
        texts = [t for t in ds["text"] if t]
        ids = []
        for i in range(0, len(texts), 4096):
            for e in enc.encode_ordinary_batch(texts[i:i + 4096]):
                ids.extend(e)
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(path)
        print(f"{split}: {arr.size:,} tokens -> {path}")


def get_batch(data, seq_len, batch_size, device):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_lengths(model, data, device, lengths=EVAL_LENGTHS, max_windows=50):
    model.eval()
    out = {}
    for L in lengths:
        if L > model.max_seq_len:
            out[L] = None
            print(f"L={L}: skipped (model max_seq_len={model.max_seq_len})")
            continue
        losses = []
        n = min((len(data) - 1) // L, max_windows)
        for w in range(n):
            i = w * L
            x = torch.from_numpy(data[i:i + L].astype(np.int64))[None].to(device)
            y = torch.from_numpy(data[i + 1:i + 1 + L].astype(np.int64))[None].to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[L] = math.exp(sum(losses) / len(losses))
        print(f"L={L}: ppl {out[L]:.2f}")
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", choices=["learned", "alibi", "rope", "none"], required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--eval-max-len", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prepare(args.data_dir)
    train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    max_seq_len = args.seq_len if args.pos == "learned" else args.eval_max_len
    model = GPT(VOCAB_SIZE, args.d_model, args.n_heads, args.n_layers, args.d_ff,
                max_seq_len=max_seq_len, pos_encoding=args.pos).to(device)
    print(f"pos={args.pos} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project="positional-encodings", name=args.pos, config=vars(args))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.1 * args.lr + 0.5 * (0.9 * args.lr) * (1 + math.cos(math.pi * p))

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" \
        else torch.autocast(device_type="cpu", enabled=False)

    model.train()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(train_data, args.seq_len, args.batch_size, device)
        with autocast:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if step % args.log_every == 0:
            lv = loss.item()
            print(f"step {step} loss {lv:.3f} ppl {math.exp(min(lv, 20)):.1f}")
            if run:
                run.log({"train/loss": lv, "train/ppl": math.exp(min(lv, 20)), "lr": lr_at(step)}, step=step)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{args.pos}.pt")
    json_path = os.path.join(args.out_dir, f"{args.pos}_extrapolation.json")
    torch.save({"model": model.state_dict(), "pos": args.pos, "args": vars(args)}, ckpt_path)

    results = eval_lengths(model, val_data, device)
    with open(json_path, "w") as f:
        json.dump({"pos": args.pos, "train_len": args.seq_len, "ppl_by_length": results}, f, indent=2)

    if run:
        run.summary["ppl_by_length"] = {str(k): v for k, v in results.items()}
        table = wandb.Table(columns=["length", "ppl"],
                            data=[[L, p] for L, p in results.items() if p is not None])
        run.log({"extrapolation": table})
        run.finish()


if __name__ == "__main__":
    main()
