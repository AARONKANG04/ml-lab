import argparse
import json
import math
import os
import time

import numpy as np
import torch

from model import GPT

VOCAB_SIZE = 50257


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
def val_loss(model, data, seq_len, device, n_windows=64, batch_size=16):
    model.eval()
    n = min(n_windows, (len(data) - 1) // seq_len)
    total = 0.0
    for start in range(0, n, batch_size):
        xs, ys = [], []
        for w in range(start, min(start + batch_size, n)):
            i = w * seq_len
            xs.append(torch.from_numpy(data[i:i + seq_len].astype(np.int64)))
            ys.append(torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64)))
        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)
        _, loss = model(x, y)
        total += loss.item() * x.size(0)
    model.train()
    return total / n


def entropy(load, n_experts):
    p = [max(v, 1e-9) for v in load]
    return -sum(v * math.log(v) for v in p) / math.log(n_experts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--capacity-factor", type=float, default=1.25)
    ap.add_argument("--name", default=None)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--seq-len", type=int, default=512)
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
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    name = args.name or ("dense" if args.experts == 0 else
                         f"switch{args.experts}" + ("-noaux" if args.alpha == 0 else ""))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prepare(args.data_dir)
    train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    model = GPT(VOCAB_SIZE, args.d_model, args.n_heads, args.n_layers, args.d_ff,
                max_seq_len=args.seq_len, n_experts=args.experts,
                capacity_factor=args.capacity_factor).to(device)
    params_total = sum(p.numel() for p in model.parameters())
    expert_extra = sum(sum(p.numel() for p in m.experts[0].parameters()) * (args.experts - 1)
                       for _, m in model.moe_layers())
    params_active = params_total - expert_extra
    print(f"{name}: {params_total/1e6:.1f}M params, {params_active/1e6:.1f}M active per token")

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project="switch-transformer", name=name, config=vars(args))

    os.makedirs("metrics", exist_ok=True)
    mf = open(os.path.join("metrics", f"{name}.jsonl"), "a")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.1 * args.lr + 0.5 * (0.9 * args.lr) * (1 + math.cos(math.pi * p))

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" \
        else torch.autocast(device_type="cpu", enabled=False)

    model.train()
    t0 = time.time()
    t_last = t0
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(train_data, args.seq_len, args.batch_size, device)
        with autocast:
            _, ce = model(x, y)
            loss = ce + args.alpha * model.aux_loss() if args.experts else ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_every == 0:
            now = time.time()
            sps = args.log_every / (now - t_last)
            t_last = now
            lv = ce.item()
            rec = {"step": step, "loss": lv, "sps": round(sps, 2)}
            wl = {"train/loss": lv, "train/ppl": math.exp(min(lv, 20)), "lr": lr_at(step), "sps": sps}
            if args.experts:
                moe = []
                for i, m in model.moe_layers():
                    load = m.load.cpu().tolist()
                    moe.append({"layer": i, "load": [round(v, 4) for v in load],
                                "drop": round(m.drop_frac, 4), "aux": round(m.aux_loss.item(), 4)})
                rec["moe"] = moe
                wl["moe/aux"] = sum(l["aux"] for l in moe) / len(moe)
                wl["moe/drop"] = sum(l["drop"] for l in moe) / len(moe)
                wl["moe/entropy"] = sum(entropy(l["load"], args.experts) for l in moe) / len(moe)
            print(f"step {step} loss {lv:.3f} {sps:.1f} steps/s")
            if step % args.eval_every == 0:
                vl = val_loss(model, val_data, args.seq_len, device)
                rec["val_loss"] = round(vl, 4)
                wl["val/loss"] = vl
                wl["val/ppl"] = math.exp(vl)
                print(f"step {step} val loss {vl:.3f} ppl {math.exp(vl):.1f}")
            if run:
                run.log(wl, step=step)
            mf.write(json.dumps(rec) + "\n")
            mf.flush()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "name": name},
               os.path.join(args.out_dir, f"{name}.pt"))

    final_vl = val_loss(model, val_data, args.seq_len, device, n_windows=400)
    result = {"name": name, "params_total": params_total, "params_active": params_active,
              "final_val_loss": round(final_vl, 4), "final_val_ppl": round(math.exp(final_vl), 2),
              "steps": args.steps, "wall_s": round(time.time() - t0, 1)}
    with open(os.path.join(args.out_dir, f"{name}_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))

    if run:
        run.summary.update(result)
        run.finish()
    mf.close()


if __name__ == "__main__":
    main()
