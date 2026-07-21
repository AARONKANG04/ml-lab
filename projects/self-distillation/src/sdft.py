"""The SDFT objective.

Three pieces, and the middle one is where all the bugs live:

1. the two prompts. The student sees a bare question. The teacher sees the same
   question with the source passage and a gold answer pasted into the prompt.
   Same weights, different context.
2. lining up the logits. Both models score the *same* response tokens, but their
   prompts have different lengths, so the positions do not match.
3. the KL between the two distributions, averaged over response tokens.
"""

import copy

import torch

# Two prompt styles, and the difference between them turned out to matter.
#
# "short" was my first design: force one-sentence answers, cap rollouts at 96
# tokens. Tidy to grade, and it strangles SDFT, because SDFT's entire training
# signal is the teacher's distribution over its own natural responses, and a
# one-line format leaves that distribution nowhere to live. SFT does not care,
# it just imitates the short gold string either way.
#
# "natural" matches the paper: no style directive, the teacher demonstrates
# "including the thinking process", generous token budget.
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
    """What the deployed model sees: the question and nothing else."""
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": STYLES[style]["sys"]},
         {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )


def teacher_prompt(tokenizer, question, context, answer, style="natural"):
    """The privileged view. Same model, but the passage it needs and an example
    answer are sitting right there in the prompt, so in-context learning does the
    work that the student has to do from weights alone.

    Follows the template from the paper: question, then a demonstration, then a
    request for the model's own answer.
    """
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
    """Right-pad token id lists into a tensor plus attention mask."""
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


def response_logits(model, tokenizer, prompt_ids, resp_ids, use_grad=False):
    """Logits over the response tokens only.

    This is the part that is easy to get quietly wrong. The student and teacher
    see different prompts but the same response. A response token at index j sits
    at absolute position prompt_len + j, and the logit that predicts it is at
    prompt_len + j - 1. Gathering on that offset is the whole trick.

    If you get the offset wrong the loss still runs and still goes down, it just
    trains on garbage, so there is a test for this in tests/test_alignment.py.
    """
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
    """The teacher is a frozen copy of the student that we later drag along with
    an EMA. It is not the pretrained model held fixed, it tracks training.
    """
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


@torch.no_grad()
def ema_update(teacher, student, alpha):
    """phi <- alpha * theta + (1 - alpha) * phi."""
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.mul_(1.0 - alpha).add_(sp.detach(), alpha=alpha)


def kl_per_token(student_logits, teacher_logits, mask, direction="forward"):
    """Per-token KL, averaged over response tokens.

    Worth knowing: the paper writes the objective as reverse KL,
    D_KL(student || teacher), but the authors' released code says every result
    was produced with per-token *forward* KL, the GKD formulation. Forward is the
    default here for that reason, and `direction` lets you check the other one.
    """
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
    """Cut padding after the first EOS but keep the EOS, so the model still
    learns where to stop."""
    out = []
    for t in ids:
        out.append(t)
        if t == eos_id:
            break
    return out or [eos_id]


@torch.no_grad()
def rollout(model, tokenizer, prompts, max_new_tokens=96, temperature=1.0):
    """Sample from the student. This is the on-policy part: the model trains on
    its own outputs, not on expert text, which is what SFT does instead.
    """
    device = next(model.parameters()).device
    model.eval()
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    add_special_tokens=False).to(device)
    gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=temperature, top_p=0.95,
                         pad_token_id=tokenizer.pad_token_id)
    start = enc["input_ids"].shape[1]
    return [trim_response(row[start:].tolist(), tokenizer.eos_token_id) for row in gen]
