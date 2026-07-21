import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sdft import response_logits, student_prompt, teacher_prompt  # noqa: E402
from train import load_model  # noqa: E402

QUESTIONS = [
    "What is the capital of France, and roughly how many people live there?",
    "Name two planets in the solar system that have rings.",
    "Who wrote Pride and Prejudice, and in what decade was it published?",
]


def test_alignment(model_name="Qwen/Qwen2.5-3B-Instruct", tol=0.85):
    model, tokenizer = load_model(model_name)
    hits = {0: 0, 1: 0, -1: 0}
    total = 0

    for question in QUESTIONS:
        prompt = student_prompt(tokenizer, question)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=40, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        resp = gen[0, enc["input_ids"].shape[1]:].tolist()

        logits = model(input_ids=torch.tensor([prompt_ids + resp], device=model.device)).logits[0]
        p = len(prompt_ids)
        for shift in hits:
            idx = [p + j - 1 + shift for j in range(len(resp))]
            if min(idx) < 0 or max(idx) >= logits.shape[0]:
                continue
            argmax = logits[idx].argmax(-1).tolist()
            hits[shift] += sum(a == b for a, b in zip(argmax, resp))
        total += len(resp)

    ours, plus, minus = hits[0] / total, hits[1] / total, hits[-1] / total
    print(f"{total} greedy tokens")
    print(f"  our alignment (p+j-1): {ours:.1%}")
    print(f"  shifted +1           : {plus:.1%}")
    print(f"  shifted -1           : {minus:.1%}")

    assert ours > tol, f"alignment looks broken, only {ours:.1%} reproduced"
    assert ours > 10 * max(plus, minus), "a shifted alignment did suspiciously well"
    print("PASS")


def test_prompts_differ_but_share_response(model_name="Qwen/Qwen2.5-3B-Instruct"):
    model, tokenizer = load_model(model_name)
    q, ctx, ans = "When did it happen?", "It happened on 3 March 2025." * 20, "3 March 2025"

    s_ids = tokenizer(student_prompt(tokenizer, q), add_special_tokens=False)["input_ids"]
    t_ids = tokenizer(teacher_prompt(tokenizer, q, ctx, ans),
                      add_special_tokens=False)["input_ids"]
    assert len(t_ids) > len(s_ids)

    resp = tokenizer(ans, add_special_tokens=False)["input_ids"]
    s_logits, s_mask = response_logits(model, tokenizer, [s_ids], [resp])
    t_logits, t_mask = response_logits(model, tokenizer, [t_ids], [resp])

    assert s_logits.shape == t_logits.shape, (s_logits.shape, t_logits.shape)
    assert torch.equal(s_mask, t_mask)
    print(f"PASS  both give {tuple(s_logits.shape)} despite prompts of "
          f"{len(s_ids)} vs {len(t_ids)} tokens")


if __name__ == "__main__":
    test_alignment()
    test_prompts_differ_but_share_response()
