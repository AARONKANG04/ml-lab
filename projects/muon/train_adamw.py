import argparse
import os
import math
import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import GPT

parser = argparse.ArgumentParser(description="Train GPT with AdamW on FineWeb-Edu")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--warmup-steps", type=int, default=200)
parser.add_argument("--max-steps", type=int, default=10000)
parser.add_argument("--grad-accum-steps", type=int, default=1)
parser.add_argument("--log-every", type=int, default=100)
parser.add_argument("--eval-every", type=int, default=500)
parser.add_argument("--eval-batches", type=int, default=50)
parser.add_argument("--save-every", type=int, default=2000)
parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
parser.add_argument("--compile", action="store_true")
args = parser.parse_args()

if args.device == "auto":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = torch.device(args.device)

print(f"Device: {device}")
torch.set_float32_matmul_precision("high")


# Data
def load_tokens(path):
    return np.memmap(path, dtype=np.uint16, mode="r")

train_data = load_tokens("../data/fineweb_edu/train.bin")
val_data = load_tokens("../data/fineweb_edu/val.bin")

seq_len = 256

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - seq_len - 1, (args.batch_size,)).numpy()
    x = np.stack([data[i:i+seq_len] for i in ix]).astype(np.int64)
    y = np.stack([data[i+1:i+1+seq_len] for i in ix]).astype(np.int64)
    x = torch.from_numpy(x).pin_memory().to(device, non_blocking=True)
    y = torch.from_numpy(y).pin_memory().to(device, non_blocking=True)
    return x, y


# LR Schedule

def get_lr(step):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    decay_ratio = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))


# Model & Optimizer

model = GPT().to(device)
n_params = sum(p.numel() for p in model.parameters()) - model.tok_emb.weight.numel()
print(f"Parameters: {n_params / 1e6:.1f}M (unique, tied embeddings)")

if args.compile:
    model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

enc = tiktoken.get_encoding("gpt2")
sample_prompt = "The theory of general relativity"
sample_tokens = enc.encode(sample_prompt)

@torch.no_grad()
def generate_sample(max_new_tokens=250):
    model.eval()
    idx = torch.tensor([sample_tokens], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.seq_len:]
        logits, _ = model(idx_cond)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=1)
    return enc.decode(idx[0].tolist())


# Training

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

train_losses = []
val_losses = []
running_loss = 0.0
running_count = 0

pbar = tqdm(range(args.max_steps), desc="AdamW")
for step in pbar:
    model.train()

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Gradient accumulation
    optimizer.zero_grad()
    accum_loss = 0.0
    for micro_step in range(args.grad_accum_steps):
        x, y = get_batch("train")
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, loss = model(x, y)
        loss = loss / args.grad_accum_steps
        loss.backward()
        accum_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    running_loss += accum_loss
    running_count += 1

    if (step + 1) % args.log_every == 0:
        avg = running_loss / running_count
        train_losses.append((step + 1, avg))
        pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}")
        running_loss = 0.0
        running_count = 0

    if (step + 1) % args.eval_every == 0:
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for _ in range(args.eval_batches):
                x, y = get_batch("val")
                with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    _, loss = model(x, y)
                total_loss += loss.item()
        avg_val = total_loss / args.eval_batches
        val_losses.append((step + 1, avg_val))
        print(f"\nStep {step+1} | val loss: {avg_val:.4f} | ppl: {math.exp(avg_val):.1f}")

    if (step + 1) % args.save_every == 0:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step + 1,
            "args": vars(args),
        }, "models/adamw.pt")
        text = generate_sample()
        print(f"\n--- Sample at step {step+1} ---\n{text}\n---")

np.save("logs/adamw_loss.npy", np.array(train_losses))
np.save("logs/adamw_val_loss.npy", np.array(val_losses))
print(f"\nSaved logs/adamw_loss.npy ({len(train_losses)} entries)")
print(f"Saved logs/adamw_val_loss.npy ({len(val_losses)} entries)")
