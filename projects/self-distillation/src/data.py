import argparse
import concurrent.futures
import json
import os
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

WIKI_UA = "ml-lab-sdft-repro/0.1 (research)"
DEEPSEEK_MODEL = "deepseek-v4-flash"
CHUNK_CHARS = 6000

ARTICLE_TITLES = [
    "2025 Myanmar earthquake",
    "2025 Kamchatka earthquake",
    "2025 Cebu earthquake",
    "Hurricane Melissa",
    "Hurricane Erin (2025)",
    "July 2025 Central Texas floods",
    "2025 Pakistan floods",
    "Typhoon Fung-wong (2025)",
    "2025 Canadian wildfires",
]

PAIR_PROMPT = """Read the passage and pick {n} specific facts worth testing.

For EACH fact write:
  - "question_a" and "question_b": two questions asking for the SAME information,
    worded as differently as you can. Different sentence structure, different
    entry point into the fact. Not a trivial rewording.
  - "answer": the short factual answer, a few words. Identical for both questions.

Rules:
- Each question must be self-contained: name the event, place and year, because
  the reader will NOT have the passage.
- Prefer facts with concrete specifics (names, numbers, dates).
- Do not invent anything absent from the passage.

Return ONLY a JSON array:
[{{"question_a": "...", "question_b": "...", "answer": "..."}}]

ARTICLE: {title}
---
{chunk}
---"""

INDIRECT_PROMPT = """Write {n} questions about this article whose answers depend on
facts in it, but which do NOT mention or quote the article. They should read like
general questions somebody might ask about the event.

Each question must name the event, place and year so it stands alone. Answers must
be short and factual.

Return ONLY a JSON array: [{{"question": "...", "answer": "..."}}]

ARTICLE: {title}
---
{chunk}
---"""


def wiki_extract(title):
    params = {
        "action": "query", "prop": "extracts", "explaintext": "1",
        "titles": title, "format": "json", "redirects": "1",
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": WIKI_UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    page = next(iter(data["query"]["pages"].values()))
    return page["title"], page.get("extract", "")


def deepseek_chat(messages, temperature=0.6, max_tokens=8000, retries=4):
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    payload = json.dumps({
        "model": DEEPSEEK_MODEL, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://api.deepseek.com/chat/completions", data=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read().decode())
            return body["choices"][0]["message"]["content"]
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"deepseek failed after {retries} tries: {last}")


def extract_json_array(text):
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    match = re.search(r"\[.*\]", cleaned, re.S)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []


def gen_pairs(title, chunk, n=12):
    text = deepseek_chat([{"role": "user", "content": PAIR_PROMPT.format(
        n=n, title=title, chunk=chunk)}])
    out = []
    for r in extract_json_array(text):
        if not isinstance(r, dict):
            continue
        qa, qb, ans = r.get("question_a"), r.get("question_b"), r.get("answer")
        if qa and qb and ans:
            out.append({"article": title, "context": chunk,
                        "question_a": str(qa).strip(), "question_b": str(qb).strip(),
                        "answer": str(ans).strip()})
    return out


def gen_indirect(title, chunk, n=12):
    text = deepseek_chat([{"role": "user", "content": INDIRECT_PROMPT.format(
        n=n, title=title, chunk=chunk)}])
    return [{"article": title, "context": chunk, "kind": "indirect",
             "question": str(r["question"]).strip(), "answer": str(r["answer"]).strip()}
            for r in extract_json_array(text)
            if isinstance(r, dict) and r.get("question") and r.get("answer")]


def fetch_corpus(out_dir):
    path = out_dir / "corpus.json"
    if path.exists():
        return json.loads(path.read_text())
    corpus = []
    for title in ARTICLE_TITLES:
        resolved, text = wiki_extract(title)
        corpus.append({"title": resolved, "text": text})
        time.sleep(0.5)
    path.write_text(json.dumps(corpus, indent=2))
    return corpus


def build(out_dir="assets", pairs_per_chunk=12, indirect_per_article=12,
          seed=0, workers=8):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = fetch_corpus(out_dir)
    print(f"corpus: {len(corpus)} articles, {sum(len(a['text']) for a in corpus):,} chars")

    pairs_path = out_dir / "qa_pairs.json"
    if pairs_path.exists():
        pairs = json.loads(pairs_path.read_text())
    else:
        jobs = [(a["title"], a["text"][i:i + CHUNK_CHARS])
                for a in corpus for i in range(0, len(a["text"]), CHUNK_CHARS)]
        pairs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(gen_pairs, t, c, pairs_per_chunk) for t, c in jobs]
            for done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                try:
                    pairs.extend(fut.result())
                except Exception as exc:
                    print(f"  chunk failed: {exc}")
                if done % 10 == 0:
                    print(f"  {done}/{len(jobs)} chunks, {len(pairs)} fact pairs")
        pairs_path.write_text(json.dumps(pairs, indent=2))

    ood_path = out_dir / "qa_test_ood.json"
    if not ood_path.exists():
        ood = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(gen_indirect, a["title"], a["text"][:12000],
                                indirect_per_article) for a in corpus]
            for fut in concurrent.futures.as_completed(futs):
                try:
                    ood.extend(fut.result())
                except Exception as exc:
                    print(f"  indirect failed: {exc}")
        ood_path.write_text(json.dumps(ood, indent=2))

    random.Random(seed).shuffle(pairs)
    train = [{"article": p["article"], "context": p["context"],
              "question": p["question_a"], "answer": p["answer"]} for p in pairs]
    test = [{"article": p["article"], "context": p["context"],
             "question": p["question_b"], "answer": p["answer"]} for p in pairs]

    (out_dir / "qa_train.json").write_text(json.dumps(train, indent=2))
    (out_dir / "qa_test_paraphrase.json").write_text(json.dumps(test, indent=2))
    print(f"{len(pairs)} fact pairs -> {len(train)} train / {len(test)} paraphrase test")
    print(f"ood: {len(json.loads(ood_path.read_text()))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.out, seed=args.seed)
