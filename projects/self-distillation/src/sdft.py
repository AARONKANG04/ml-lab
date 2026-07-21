import copy

import torch

STYLES = {
    "short": {
        "sys": ("You are answering factual questions about recent events. "
                "Reply with the answer only, in one short sentence."),
        "closing": "Now answer with a response of your own:",
        "max_new_tokens": 96,
    },
    "natural": {
        "sys": "You are a helpful assistant.",
        "closing": "Now answer with a response of your own, including the thinking process:",
        "max_new_tokens": 256,
    },
}


def student_prompt(tokenizer, question, style="natural"):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": STYLES[style]["sys"]},
         {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )


def teacher_prompt(tokenizer, question, context, answer, style="natural"):
    demo = (
        f"{question}\n\n"
        "This is an example for a response to the question:\n\n"
        f"Source passage:\n{context.strip()}\n\n"
        f"Answer: {answer.strip()}\n\n"
        f"{STYLES[style]['closing']}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": STYLES[style]["sys"]},
         {"role": "user", "content": demo}],
        tokenize=False, add_generation_prompt=True,
    )


def pad_batch(seqs, pad_id, device):
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


def response_logits(model, tokenizer, prompt_ids, resp_ids, use_grad=False):
    device = next(model.parameters()).device
    seqs = [p + r for p, r in zip(prompt_ids, resp_ids)]
    ids, mask = pad_batch(seqs, tokenizer.pad_token_id, device)

    with torch.enable_grad() if use_grad else torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits

    width = max(len(r) for r in resp_ids)
    gather_idx = torch.zeros((len(resp_ids), width), dtype=torch.long)
    resp_mask = torch.zeros((len(resp_ids), width), dtype=torch.bool)
    for i, (p, r) in enumerate(zip(prompt_ids, resp_ids)):
        gather_idx[i, : len(r)] = torch.arange(len(p) - 1, len(p) + len(r) - 1)
        resp_mask[i, : len(r)] = True
    gather_idx = gather_idx.to(device)

    picked = logits.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, logits.size(-1)))
    return picked, resp_mask.to(device)


def make_teacher(student):
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


@torch.no_grad()
def ema_update(teacher, student, alpha):
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.mul_(1.0 - alpha).add_(sp.detach(), alpha=alpha)


def kl_per_token(student_logits, teacher_logits, mask, direction="forward"):
    s_logp = torch.log_softmax(student_logits.float(), dim=-1)
    t_logp = torch.log_softmax(teacher_logits.float(), dim=-1)
    if direction == "forward":
        kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1)
    elif direction == "reverse":
        kl = (s_logp.exp() * (s_logp - t_logp)).sum(-1)
    else:
        raise ValueError(f"unknown direction {direction!r}")
    return (kl * mask).sum() / mask.sum().clamp(min=1)


def trim_response(ids, eos_id):
    out = []
    for t in ids:
        out.append(t)
        if t == eos_id:
            break
    return out or [eos_id]


@torch.no_grad()
def rollout(model, tokenizer, prompts, max_new_tokens=256, temperature=1.0):
    device = next(model.parameters()).device
    model.eval()
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    add_special_tokens=False).to(device)
    gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=temperature, top_p=0.95,
                         pad_token_id=tokenizer.pad_token_id)
    start = enc["input_ids"].shape[1]
    return [trim_response(row[start:].tolist(), tokenizer.eos_token_id) for row in gen]
