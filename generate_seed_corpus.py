import random, json, textstat, pandas as pd, string
from datasets import load_dataset
from collections import Counter, defaultdict

race = load_dataset("race", "high")

df = pd.DataFrame(race["train"])

from collections import defaultdict

#creating a dictionary of source passages (articles), questions, options, and answers
def regroup_by_passage(hf_split):
    grouped = defaultdict(lambda: {"article": None, "questions":[], "options":[], "answers": []})
    for row in hf_split:
        passage_id = row["example_id"]
        if grouped[passage_id]["article"] is None:
            grouped[passage_id]["article"] = row["article"]
        elif grouped[passage_id]["article"] != row["article"]:
            print(f"WARNING: article mismatch under id {passage_id}") # id collision, shouldn't happen
        grouped[passage_id]["questions"].append(row["question"])
        grouped[passage_id]["options"].append(row["options"])
        grouped[passage_id]["answers"].append(row["answer"])
    return grouped

train_passages = regroup_by_passage(race["train"])
val_passages = regroup_by_passage(race["validation"])
test_passages = regroup_by_passage(race["test"])

print(f"train: {len(race['train'])} rows -> {len(train_passages)} passages")
print(f"val: {len(race['validation'])} rows -> {len(val_passages)} passages")
print(f"test: {len(race['test'])} rows -> {len(test_passages)} passages")

# Removing duplicates from all three passages. 
# Methodology -> combine all three splits into one and run deduplication function.

import re, hashlib

# Normalizing the source passages to have same casing, removing
# punctuation 
def normalize(text):
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text) # removes trailing white spaces
    text = text.translate(str.maketrans("", "", string.punctuation)) # remove punctuation
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"') # remove quote marks(',") begining and ending
    return text

all_passages = []
for split_name, passages in [("train", train_passages), ("validation", val_passages), ("test", test_passages)]:
    for passage_id, data in passages.items():
        all_passages.append({
            "passage_id": passage_id,
            "source_split": split_name,
            "article": data["article"],
            "questions": data["questions"],
            "options": data["options"],
            "answers": data["answers"]
        })

print(f"Combined pool before dedup: {len(all_passages)} passages")

seen_hashes = {}
deduped =[]
dup_log = []

for rec in all_passages:
    h = hashlib.md5(normalize(rec["article"]).encode()).hexdigest()
    if h in seen_hashes:
        dup_log.append((rec["passage_id"], seen_hashes[h]))
        continue
    seen_hashes[h] = rec["passage_id"]
    deduped.append(rec)

print(f"Removed {len(dup_log)} exact-duplicate passages")
print(f"Remaining unique passages: {len(deduped)}")

if dup_log:
    print("Sample duplicate pairs (dup_id, kept_id):", dup_log)


# Removing underlined references

UNDERLINE_PATTERN = re.compile(r"\bunderlin(e|ed|ing)\b", re.IGNORECASE)

MIN_QUESTIONS = 2 # if questions remaining are less than 2, drop the passage

total_q_before = 0 # how many questions existed before filtering
total_q_removed = 0 # how many got removed for referencing "underlined" text
passages_dropped = 0 
cleaned = [] 

for rec in deduped:

    total_q_before += len(rec["questions"])

    keep_index = [i for i, q in enumerate(rec["questions"]) if not UNDERLINE_PATTERN.search(q)]

    removed = len(rec["questions"]) - len(keep_index)
    total_q_removed += removed

    rec["questions"] = [rec["questions"][i] for i in keep_index]
    rec["options"] = [rec["options"][i] for i in keep_index]
    rec["answers"] = [rec["answers"][i] for i in keep_index]

    if len(rec["questions"]) >= MIN_QUESTIONS:
        cleaned.append(rec)
    else:
        passages_dropped += 1

print(f"Questions before filter: {total_q_before}")
print(f"Underline referencing questions removed: {total_q_removed}")
print(f"Passages dropped: {passages_dropped}")
print(f"Passages remaining: {len(cleaned)}")

# Checking grade-level of dataset questions

final_rows = []

for rec in cleaned:
    article = rec["article"]

    row = {
        "passage_id": rec["passage_id"],
        "source_split": rec["source_split"],
        "word_count": len(article.split()),

        # Flesch-Kincaid grade level estimate for this specific passage 
        "flesh_kincaid_grade": round(textstat.flesch_kincaid_grade(article), 1),
        "num_gold_questions_retained": len(rec["questions"]),
        "article": article,
        "questions": rec["questions"],
        "options": rec["options"],
        "answers": rec["answers"],
        # Licensing note
        "license_note": "RACE (Lai et al. 2017) - non-commerical research use only, no formal license grant",
        "contamination_flag": None
    }
    final_rows.append(row)

print(f"{len(final_rows)} final corpus rows")

grade_buckets = Counter(int(row["flesh_kincaid_grade"]) for row in final_rows)
for grade in sorted(grade_buckets):
    print(f"Grade {grade}: {grade_buckets[grade]} passages")

# Creating the corpus with grade 9 - 12 passages

grade_9_12 = [row for row in final_rows if 9 <= int(row["flesh_kincaid_grade"]) <= 12]
print(f"Passages in grade 9-12: {len(grade_9_12)}")

by_grade = defaultdict(list)
for row in grade_9_12:
    by_grade[int(row["flesh_kincaid_grade"])].append(row)

for g in sorted(by_grade):
    print(f" Grade {g}: {len(by_grade[g])} candidates")


def write_json1(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

write_json1(grade_9_12, "seed_corpus.json1")