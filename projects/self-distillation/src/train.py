import argparse
import json
import math
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sdft import (
    STYLES,
    ema_update,
    kl_per_token,
    make_teacher,
    response_logits,
    rollout,
    student_prompt,
    teacher_prompt,
)


def load_model(model_name, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    return model, tokenizer


def make_optimizer(model, lr, total_steps, warmup=10):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, prog)))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def train_sdft(student, teacher, tokenizer, data, epochs=4, lr=5e-5, batch_size=8,
               alpha=0.02, temperature=1.0, style="natural", max_new_tokens=None,
               direction="forward", seed=0, log_every=10):
    if max_new_tokens is None:
        max_new_tokens = STYLES[style]["max_new_tokens"]
    steps_per_epoch = math.ceil(len(data) / batch_size)
    opt, sched = make_optimizer(student, lr, epochs * steps_per_epoch)
    rng = random.Random(seed)
    history, step = [], 0

    for epoch in range(epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            batch = [data[j] for j in order[i:i + batch_size]]
            s_prompts = [student_prompt(tokenizer, it["question"], style) for it in batch]

            student.config.use_cache = True
            resp = rollout(student, tokenizer, s_prompts, max_new_tokens, temperature)

            s_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in s_prompts]
            t_ids = [
                tokenizer(
                    teacher_prompt(tokenizer, it["question"], it["context"],
                                   it["answer"], style),
                    add_special_tokens=False,
                )["input_ids"]
                for it in batch
            ]

            t_logits, mask = response_logits(teacher, tokenizer, t_ids, resp, use_grad=False)

            student.train()
            student.config.use_cache = False
            s_logits, _ = response_logits(student, tokenizer, s_ids, resp, use_grad=True)

            loss = kl_per_token(s_logits, t_logits, mask, direction)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            ema_update(teacher, student, alpha)

            del s_logits, t_logits
            history.append({"step": step, "epoch": epoch, "loss": loss.detach().item()})
            if step % log_every == 0:
                print(f"  sdft step {step:4d} ep{epoch} kl {loss.detach().item():.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e}")
            step += 1

    student.config.use_cache = True
    return history


def train_sft(student, tokenizer, data, epochs=4, lr=5e-5, batch_size=8,
              style="natural", seed=0, log_every=10):
    steps_per_epoch = math.ceil(len(data) / batch_size)
    opt, sched = make_optimizer(student, lr, epochs * steps_per_epoch)
    rng = random.Random(seed)
    history, step = [], 0
    device = next(student.parameters()).device

    for epoch in range(epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            batch = [data[j] for j in order[i:i + batch_size]]
            s_ids = [
                tokenizer(student_prompt(tokenizer, it["question"], style),
                          add_special_tokens=False)["input_ids"]
                for it in batch
            ]
            gold = [
                tokenizer(it["answer"], add_special_tokens=False)["input_ids"]
                + [tokenizer.eos_token_id]
                for it in batch
            ]

            student.train()
            student.config.use_cache = False
            logits, mask = response_logits(student, tokenizer, s_ids, gold, use_grad=True)

            targets = torch.full(mask.shape, -100, dtype=torch.long, device=device)
            for r, g in enumerate(gold):
                targets[r, : len(g)] = torch.tensor(g, device=device)
            loss = torch.nn.functional.cross_entropy(
                logits.float().flatten(0, 1), targets.flatten(), ignore_index=-100)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            del logits
            history.append({"step": step, "epoch": epoch, "loss": loss.detach().item()})
            if step % log_every == 0:
                print(f"  sft  step {step:4d} ep{epoch} ce {loss.detach().item():.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e}")
            step += 1

    student.config.use_cache = True
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["sdft", "sft"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default="assets/qa_train.json")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--style", default="natural", choices=["natural", "short"])
    ap.add_argument("--direction", default="forward", choices=["forward", "reverse"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = json.loads(Path(args.data).read_text())
    student, tokenizer = load_model(args.model)

    if args.method == "sdft":
        teacher = make_teacher(student)
        history = train_sdft(
            student, teacher, tokenizer, data, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, alpha=args.alpha, style=args.style,
            temperature=args.temperature, direction=args.direction, seed=args.seed,
        )
    else:
        history = train_sft(
            student, tokenizer, data, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, style=args.style, seed=args.seed,
        )

    out = Path(args.out) / f"{args.method}-{args.style}"
    out.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
