import json, os, random, re, sys, time, urllib.request, urllib.error
from collections import Counter, defaultdict

# Week 3 baseline: passage in, N RACE-format MCQs out, zero-shot vs few-shot.
#   python generate.py         run (resumable: skips rows already in OUT_PATH)
#   python generate.py stats   summary table for the prompt comparison
#   python generate.py test    offline self-check

CORPUS_PATH = "seed_corpus.json1"
OUT_PATH = "generations.jsonl"
SEED = 42
N_QUESTIONS = 4
PER_GRADE = 5          # 5 passages x grades 9-12 = 20 passages
N_SHOTS = 2
TEMPERATURE = 0.7
PROMPT_VERSION = "v1"
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Extra request params per model. gpt-oss can't turn reasoning off, so it runs at the lowest level;
# qwen's reasoning is off by default; allam has no reasoning option and a 4k context window.
MODELS = {
    "openai/gpt-oss-20b": {"reasoning_effort": "low"},
    "openai/gpt-oss-120b": {"reasoning_effort": "low"},
    "qwen/qwen3.8-27b": {"reasoning_effort": "none"},
    "allam-2-7b": {},
}

SYSTEM = "You write multiple-choice reading comprehension questions for high school students (grades 9-12)."

INSTRUCTIONS = f"""Write exactly {N_QUESTIONS} multiple-choice questions that test comprehension of the passage below.
Each question must be answerable from the passage alone, have exactly 4 options, and exactly one correct option.
Return only JSON in this format:
{{"questions": [{{"question": "...", "options": ["...", "...", "...", "..."], "answer": "A"}}]}}
"answer" is the letter (A, B, C or D) of the correct option.

Passage:
"""


def load_corpus():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def pick_passages(corpus):
    """Few-shot exemplars and the eval sample, both fixed by SEED and disjoint."""
    rng = random.Random(SEED)
    # Short exemplars keep the few-shot prompt inside allam's 4k context.
    shot_pool = [r for r in corpus if r["word_count"] <= 250 and len(r["questions"]) >= N_QUESTIONS]
    shots = rng.sample(shot_pool, N_SHOTS)
    shot_ids = {r["passage_id"] for r in shots}
    by_grade = defaultdict(list)
    for r in corpus:
        if r["passage_id"] not in shot_ids:
            by_grade[int(r["flesh_kincaid_grade"])].append(r)
    sample = [r for g in (9, 10, 11, 12) for r in rng.sample(by_grade[g], PER_GRADE)]
    return shots, sample


def gold_json(rec):
    qs = [{"question": q, "options": o, "answer": a}
          for q, o, a in zip(rec["questions"], rec["options"], rec["answers"])][:N_QUESTIONS]
    return json.dumps({"questions": qs}, ensure_ascii=False)


def build_messages(passage, shots):
    msgs = [{"role": "system", "content": SYSTEM}]
    for s in shots:  # gold RACE items as prior turns; same user template as the target
        msgs += [{"role": "user", "content": INSTRUCTIONS + s["article"]},
                 {"role": "assistant", "content": gold_json(s)}]
    msgs.append({"role": "user", "content": INSTRUCTIONS + passage})
    return msgs


def call(model, messages):
    body = {"model": model, "messages": messages, "temperature": TEMPERATURE, "seed": SEED,
            "response_format": {"type": "json_object"}, **MODELS[model]}
    req = urllib.request.Request(API_URL, data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}", "Content-Type": "application/json",
        "User-Agent": "questgen/0.1"})
    while True:
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if e.code == 429 or e.code >= 500:
                wait = float(e.headers.get("retry-after") or 10)
                if wait > 120:  # daily cap: stop, rerun tomorrow resumes where this left off
                    sys.exit(f"Rate limited on {model} for {wait:.0f}s. Rerun later.\n{msg[:300]}")
                time.sleep(wait)
                continue
            if e.code in (400, 413):
                return {"error": msg}  # e.g. prompt too long: a real result, recorded as a failure
            sys.exit(f"HTTP {e.code} from {model}: {msg[:300]}")  # auth etc.: don't record, so a rerun retries


def validate_item(q):
    opts = q.get("options") if isinstance(q, dict) else None
    return (isinstance(q.get("question"), str) and q["question"].strip() != ""
            and isinstance(opts, list) and len(opts) == 4
            and all(isinstance(o, str) and o.strip() for o in opts)
            and len({o.strip().lower() for o in opts}) == 4
            and q.get("answer") in ("A", "B", "C", "D"))


def parse(content):
    try:
        qs = json.loads(content)["questions"]
        return qs if isinstance(qs, list) else None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def run():
    corpus = load_corpus()
    shots, sample = pick_passages(corpus)
    done = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding="utf-8") as f:
            done = {(r["passage_id"], r["model"], r["condition"]) for r in map(json.loads, f)}
    with open(OUT_PATH, "a", encoding="utf-8") as out:
        for model in MODELS:
            for condition, k in (("zero_shot", 0), ("few_shot", N_SHOTS)):
                for rec in sample:
                    key = (rec["passage_id"], model, condition)
                    if key in done:
                        continue
                    resp = call(model, build_messages(rec["article"], shots[:k]))
                    msg = resp["choices"][0]["message"] if "choices" in resp else {}
                    content = msg.get("content")
                    row = {
                        "passage_id": rec["passage_id"], "grade": int(rec["flesh_kincaid_grade"]),
                        "model": model, "condition": condition, "prompt_version": PROMPT_VERSION,
                        "params": {"temperature": TEMPERATURE, "seed": SEED, **MODELS[model]},
                        "shot_ids": [s["passage_id"] for s in shots[:k]],
                        "raw": content, "reasoning": msg.get("reasoning"),
                        "questions": parse(content) if content else None,
                        "usage": resp.get("usage"), "error": resp.get("error"),
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    n = len(row["questions"] or [])
                    print(f"{model:22} {condition:9} {rec['passage_id']:24} items={n}"
                          + (" ERROR" if row["error"] else ""))


def ngrams(text, n=4):
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def copy_rate(stem, passage):
    """Share of the stem's 4-grams that appear verbatim in the passage."""
    s = ngrams(stem)
    return len(s & ngrams(passage)) / len(s) if s else 0.0


def stats():
    articles = {r["passage_id"]: r["article"] for r in load_corpus()}
    groups = defaultdict(list)
    with open(OUT_PATH, encoding="utf-8") as f:
        for r in map(json.loads, f):
            groups[(r["model"], r["condition"])].append(r)
    print(f"{'model':22} {'cond':9} {'calls':>5} {'parsed':>6} {'exact4':>6} {'valid':>9} "
          f"{'dupes':>5} {'copy':>5} {'tokens':>6}  answer A/B/C/D")
    for (model, cond), rows in sorted(groups.items()):
        parsed = [r for r in rows if r["questions"] is not None]
        items = [(q, r["passage_id"]) for r in parsed for q in r["questions"]]
        valid = [(q, pid) for q, pid in items if validate_item(q)]
        dupes = sum(len(r["questions"]) - len({q.get("question", "").strip().lower()
                    for q in r["questions"] if isinstance(q, dict)}) for r in parsed)
        copy = sum(copy_rate(q["question"], articles[pid]) for q, pid in valid) / max(len(valid), 1)
        tokens = sum((r["usage"] or {}).get("total_tokens", 0) for r in rows) / len(rows)
        letters = Counter(q["answer"] for q, _ in valid)
        print(f"{model:22} {cond:9} {len(rows):5} {len(parsed):6} "
              f"{sum(len(r['questions']) == N_QUESTIONS for r in parsed):6} "
              f"{len(valid):4}/{len(items):<4} {dupes:5} {copy:5.2f} {tokens:6.0f}  "
              + "/".join(str(letters[c]) for c in "ABCD"))


def test():
    good = {"question": "Why?", "options": ["a", "b", "c", "d"], "answer": "B"}
    assert validate_item(good)
    assert not validate_item({**good, "answer": "E"})
    assert not validate_item({**good, "options": ["a", "b", "c"]})
    assert not validate_item({**good, "options": ["a", "A", "c", "d"]})
    assert parse('{"questions": [1]}') == [1] and parse("nope") is None and parse('{"q": 1}') is None
    assert copy_rate("the cat sat on the mat", "yesterday the cat sat on the mat") == 1.0
    assert copy_rate("why did the dog bark loudly", "the cat sat on the mat") == 0.0
    shot = {"article": "P", "questions": ["q"] * 5, "options": [["a", "b", "c", "d"]] * 5, "answers": ["A"] * 5}
    msgs = build_messages("T", [shot])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert len(json.loads(msgs[2]["content"])["questions"]) == N_QUESTIONS
    print("ok")


if __name__ == "__main__":
    {"stats": stats, "test": test}.get(sys.argv[1] if len(sys.argv) > 1 else "", run)()
