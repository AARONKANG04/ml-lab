import argparse
import concurrent.futures
import json
import re
from pathlib import Path

import torch

from data import deepseek_chat
from sdft import student_prompt, teacher_prompt

JUDGE_PROMPT = """Grade a model's answer against a reference answer.

Question: {question}
Reference answer: {gold}
Model answer: {pred}

The model answer is CORRECT if it contains the key facts of the reference answer.
Wording may differ. Extra correct detail is fine. If the question asks for several
facts, ALL of them must be present and right. Missing, vague or wrong facts, or a
refusal, are INCORRECT.

Reply with exactly one word: CORRECT or INCORRECT."""

FORGET_TASKS = {
    "mmlu": 40,
    "hellaswag": 500,
    "winogrande": 500,
    "truthfulqa_mc2": 400,
    "arc_challenge": 500,
}
_METRIC_PREF = ("acc_norm,none", "acc,none")


def judge_one(item, pred):
    reply = deepseek_chat(
        [{"role": "user", "content": JUDGE_PROMPT.format(
            question=item["question"], gold=item["answer"], pred=pred or "(no answer)")}],
        temperature=0.0, max_tokens=2000,
    )
    verdicts = re.findall(r"\b(INCORRECT|CORRECT)\b", reply.upper())
    return bool(verdicts) and verdicts[-1] == "CORRECT"


def judge_all(items, preds, workers=16):
    results = [False] * len(items)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(judge_one, it, pr): i
                for i, (it, pr) in enumerate(zip(items, preds))}
        for fut in concurrent.futures.as_completed(futs):
            try:
                results[futs[fut]] = fut.result()
            except Exception:
                results[futs[fut]] = False
    return results


@torch.no_grad()
def generate(model, tokenizer, prompts, max_new_tokens=256, batch_size=16):
    device = next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=8192).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        for seq in gen:
            out.append(tokenizer.decode(seq[enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True).strip())
    return out


def score(model, tokenizer, items, use_context=False, style="natural",
          max_new_tokens=256):
    if use_context:
        prompts = [teacher_prompt(tokenizer, it["question"], it["context"],
                                  it["answer"], style)
                   for it in items]
    else:
        prompts = [student_prompt(tokenizer, it["question"], style) for it in items]
    preds = generate(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
    ok = judge_all(items, preds)
    return sum(ok) / len(ok), preds, ok


def eval_forgetting(model, tokenizer, tasks=None):
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tasks = tasks or FORGET_TASKS
    model.eval()
    model.config.use_cache = True
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=32)
    scores = {}
    for task, limit in tasks.items():
        res = lm_eval.simple_evaluate(model=lm, tasks=[task], limit=limit, verbosity="ERROR")
        row = res["results"].get(task, {})
        for key in _METRIC_PREF:
            if key in row:
                scores[task] = float(row[key])
                break
    scores["average"] = sum(scores.values()) / len(scores)
    return scores


def evaluate_all(model, tokenizer, paraphrase, ood, trained_wording=None, label=""):
    para_acc, para_preds, para_ok = score(model, tokenizer, paraphrase)
    ood_acc, _, _ = score(model, tokenizer, ood)
    seen_acc = None
    if trained_wording:
        seen_acc, _, _ = score(model, tokenizer, trained_wording)
    forget = eval_forgetting(model, tokenizer)

    print(f"\n=== {label} ===")
    print(f"  new knowledge : paraphrase {para_acc:.1%}  ood {ood_acc:.1%}"
          + (f"  trained-wording {seen_acc:.1%}" if seen_acc is not None else ""))
    print("  prior caps    : " + "  ".join(f"{k} {v:.3f}" for k, v in forget.items()))
    return {
        "label": label,
        "knowledge": {"paraphrase": para_acc, "ood": ood_acc,
                      "trained_wording": seen_acc},
        "forgetting": forget,
        "samples": [{"q": it["question"], "gold": it["answer"], "pred": p, "ok": bool(o)}
                    for it, p, o in zip(paraphrase[:30], para_preds[:30], para_ok[:30])],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="assets")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()

    from train import load_model

    model, tokenizer = load_model(args.model)
    data = Path(args.data)
    paraphrase = json.loads((data / "qa_test_paraphrase.json").read_text())[:args.limit]
    ood = json.loads((data / "qa_test_ood.json").read_text())
    trained = json.loads((data / "qa_train.json").read_text())[:150]

    res = evaluate_all(model, tokenizer, paraphrase, ood, trained,
                       label=args.label or args.model)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
