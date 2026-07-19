"""Train the seq2seq model on WMT14 en-fr, with checkpointing and W&B logging.

    python -m src.train
    python -m src.train --max-train 2000000
    python -m src.train --resume artifacts/ckpt_latest.pt

Checkpoints are written every --ckpt-every steps and can be resumed with
--resume, so an interrupted run loses at most that many steps.
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import yaml

from .data import build_dataloaders, PAD_ID, SRC_LANG, TGT_LANG
from .model_fast import Seq2SeqFast


@torch.no_grad()
def evaluate(model, loader, device, max_batches=50):
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        src = batch["src"].to(device)
        tin = batch["tgt_in"].to(device)
        tout = batch["tgt_out"].to(device)
        logits = model(src, tin, batch["src_len"])
        total += crit(logits.reshape(-1, logits.size(-1)), tout.reshape(-1)).item()
        n += 1
    model.train()
    return total / max(n, 1)


def save_ckpt(path, model, optimizer, step, cfg):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # write to a temp file then rename, so an interrupted write can't corrupt it
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "cfg": cfg}, tmp)
    os.replace(tmp, path)


def log_artifact(run, path, step):
    if run is None:
        return
    import wandb
    art = wandb.Artifact("seq2seq-ckpt", type="model", metadata={"step": step})
    art.add_file(path)
    run.log_artifact(art, aliases=["latest", f"step-{step}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max-train", type=int, default=None, help="cap number of training pairs")
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--artifact-every", type=int, default=5000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=500000)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    loaders, (src_tok, tgt_tok) = build_dataloaders(
        out_dir=args.out_dir,
        batch_size=cfg["batch_size"],
        src_vocab_size=cfg["src_vocab_size"],
        tgt_vocab_size=cfg["tgt_vocab_size"],
        max_len=cfg["max_len"],
        max_train_samples=args.max_train,
        num_workers=0,
    )
    cfg["src_vocab_size"] = src_tok.get_vocab_size()
    cfg["tgt_vocab_size"] = tgt_tok.get_vocab_size()

    model = Seq2SeqFast(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        step = ck["step"]
        print(f"resumed from {args.resume} at step {step}")

    print(f"device={device} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"batches/epoch={len(loaders['train'])}")

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project="seq2seq-wmt14-enfr", name="cudnn-lstm-4x1000", config=cfg)
        # a checkpoint's embeddings are tied to these exact BPE ids, so keep them together
        tok_art = wandb.Artifact("tokenizers", type="tokenizer")
        for lang in (SRC_LANG, TGT_LANG):
            fp = os.path.join(args.out_dir, f"tokenizer_{lang}.json")
            if os.path.exists(fp):
                tok_art.add_file(fp)
        run.log_artifact(tok_art)

    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    model.train()
    epoch = 0
    while step < args.max_steps:
        epoch += 1
        for batch in loaders["train"]:
            src = batch["src"].to(device, non_blocking=True)
            tin = batch["tgt_in"].to(device, non_blocking=True)
            tout = batch["tgt_out"].to(device, non_blocking=True)
            t0 = time.time()
            logits = model(src, tin, batch["src_len"])
            loss = crit(logits.reshape(-1, logits.size(-1)), tout.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            step += 1

            if step % args.log_every == 0:
                ntok = (tout != PAD_ID).sum().item()
                tps = ntok / max(time.time() - t0, 1e-6)
                lv = loss.item()
                print(f"step {step} epoch {epoch} loss {lv:.3f} ppl {math.exp(min(lv,20)):.1f} {tps:.0f} tok/s")
                if run:
                    run.log({"train/loss": lv, "train/ppl": math.exp(min(lv, 20)),
                             "train/grad_norm": float(gnorm), "train/toks_per_s": tps,
                             "epoch": epoch}, step=step)

            if step % args.eval_every == 0:
                vl = evaluate(model, loaders["validation"], device)
                print(f"  [eval] step {step} val_loss {vl:.3f} val_ppl {math.exp(min(vl,20)):.1f}")
                if run:
                    run.log({"val/loss": vl, "val/ppl": math.exp(min(vl, 20))}, step=step)

            if step % args.ckpt_every == 0:
                ckpt = os.path.join(args.out_dir, "ckpt_latest.pt")
                save_ckpt(ckpt, model, optimizer, step, cfg)
                if step % args.artifact_every == 0:
                    log_artifact(run, ckpt, step)

            if step >= args.max_steps:
                break

    final = os.path.join(args.out_dir, "ckpt_final.pt")
    save_ckpt(final, model, optimizer, step, cfg)
    log_artifact(run, final, step)
    if run:
        run.finish()


if __name__ == "__main__":
    main()
