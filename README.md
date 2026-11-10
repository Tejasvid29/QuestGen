# QuestGen

Does the way we prompt an LLM change how well its generated questions test passage comprehension?
QuestGen generates multiple-choice reading comprehension questions for grades 9–12 from RACE passages
with several open-weight LLMs and compares generation methods against the RACE gold questions.

CS 180 research project, San Jose State University, Fall 2026. Small-scale by design: tens of passages, free-tier APIs.

## Repository

| File | What it is |
|---|---|
| `generate_seed_corpus.py` | Week 1: builds the seed corpus from RACE-high |
| `generate.py` | Week 3: baseline generation pipeline (zero-shot vs few-shot) |
| `week4_metrics_rubric.md` | Week 4: automatic metrics for QG, their limits, and the human quality rubric |

Data files (`seed_corpus.json1`) are not tracked: RACE is for non-commercial research use only, and the corpus is fully reproducible from the script.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install datasets textstat pandas
python generate_seed_corpus.py     # writes seed_corpus.json1
export GROQ_API_KEY=...            # free key from console.groq.com
```

`generate.py` uses only the Python standard library.

## Progress

### Week 1: Seed corpus
- **Target audience:** grades 9–12. **Source dataset:** RACE-high (Lai et al., 2017).
- **Cleaning:** removed exact-duplicate passages across splits, removed questions that refer to underlined text, and dropped passages left with fewer than 2 questions.
- **Grade filter:** assigned each passage a Flesch-Kincaid grade and kept grades 9–12.
- **Size:** about 7,050 passages (7,042 in the original build, 7,056 on the latest rebuild, probably from small upstream dataset changes).

### Week 2: Annotation schema
- Pilot guidelines for labeling RACE and generated MCQs against one acceptability standard.
- Item codes I1–I5 and distractor codes D1–D4, based on Mitkov & Ha (2003) and Kurdi et al. (2020).

### Week 3: Baseline generation pipeline
- **Input and output:** a passage in, 4 MCQs out, in RACE format: `{question, options[4], answer}`. Generated items can go straight into the blinded annotation pilot.
- **Models (Groq free tier):**

| Model | Notes |
|---|---|
| `openai/gpt-oss-20b`, `openai/gpt-oss-120b` | reasoning on at `low` (can't be disabled) |
| `qwen/qwen3.8-27b` | reasoning off |
| `allam-2-7b` | no reasoning, 4k context |

- **Conditions:** zero-shot vs 2-shot.
  - The few-shot examples are gold RACE items from 2 short passages, given as earlier chat turns.
  - Those 2 passages are excluded from the sample.
  - The instructions are identical in both conditions.
- **Sample:** 20 passages, 5 per grade 9–12, with a fixed seed (42).

```bash
python generate.py          # writes generations.jsonl; resumable, so rerun after hitting the daily token cap
python generate.py stats    # prompt-comparison table
python generate.py test     # offline self-check
```

`stats` reports these, per model and condition:
- parse rate
- valid 4-option items
- duplicate stems
- stem copying from the passage (4-gram overlap)
- tokens per call
- answer-letter distribution

### Week 4: Metrics and rubric
`week4_metrics_rubric.md` covers:
- **Metrics:** BLEU, ROUGE, METEOR, BERTScore, perplexity and QA-based evaluation, with their known limits for question generation.
- **Rubric:** a 1–3 human quality rubric (answerability, relevance, grammaticality, non-triviality), mapped to the Week 2 codes.

## References
- Lai et al. 2017, *RACE: Large-scale ReAding Comprehension Dataset From Examinations*. arXiv:1704.04683
- Kurdi et al. 2020, *A Systematic Review of Automatic Question Generation for Educational Purposes*. IJAIED
- Mitkov & Ha 2003, *Computer-Aided Generation of Multiple-Choice Tests*
