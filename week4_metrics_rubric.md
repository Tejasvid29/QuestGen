# Week 4: Metrics and Quality Rubric for Generated MCQs

**Setup.** Each generated item is `{question, options[4], answer}` for a RACE-high passage. "Reference" means the RACE gold questions for the same passage. RACE gives several gold questions per passage and none of them lines up one-to-one with a generated question, so each generated stem is scored against **every** gold stem for that passage and the **best** score is kept.

## 1. Automatic metrics

| Metric | What it measures | How we'd compute it | Known limits for QG |
|---|---|---|---|
| **BLEU-4** | n-gram *precision* against references (from MT) [S] | sacreBLEU, generated stem vs. gold stems | A passage has many valid questions, but BLEU rewards only the ones worded like the gold. It also rewards stems that copy passage text. It says nothing about the key or the distractors. |
| **ROUGE-L** | Longest-common-subsequence overlap, *recall*-oriented (from summarization) [S] | `rouge-score`, same pairing | Same as BLEU. [9] calls ROUGE "known to be limited", with poor correlation to human judgment, and finds that one reference covers little of the valid output space. |
| **METEOR** | Unigram match plus stemming and synonyms [S] | `nltk` METEOR | Softens the exact-wording problem but keeps the rest. In [9], METEOR did not correlate better than ROUGE. |
| **BERTScore** | Cosine similarity of contextual token embeddings, greedily matched into P/R/F1 [8 §3] | `bert-score` F1 (authors recommend F1, roberta-large) [8 §5, §7] | Credits paraphrases, but [8, App. A] notes it struggles to detect factual errors (changed numbers or days). A fluent question with a **wrong key** can still score high. |
| **Perplexity** | How likely a language model finds the text, i.e. fluency | Small local LM (e.g. GPT-2) on the stem | Measures fluency, not correctness, relevance, or answerability. It also depends on which LM computes it. |
| **QA-based (QuestEval-style)** | QuestEval generates questions and checks whether a QA model can answer them from the text. It needs **no reference** [9 §3–4]. | Adapted to QG: a model answers our MCQ from the passage, and we check that its answer matches the key | Built for summarization, not QG [9]. Adds model calls and inherits the QA model's errors. It checks answerability only, not quality. |

All five reference-based metrics share one gap: they compare **stems only**, so they ignore the MCQ properties that matter most here, namely a correct key and functional distractors. The survey [S] finds BLEU, ROUGE, and METEOR in just 5, 6, and 3 studies, and reports that human-based measures are the most common way QG is evaluated. Even their advocates (Wang et al. 2018, cited in [S]) claim only that these metrics are "initial, inexpensive" indicators of fluency and relevance.

## 2. Human quality rubric (aligned with the Week 2 schema v0.1)

Each criterion is scored 1–3 per item. **A score of 1 equals "the mapped schema code applies"**, so the rubric adds graded detail and never overrides the schema's binary Acceptable/Not acceptable label. Examples use the schema's school-garden passage.

| Criterion | 3 (good) | 2 (weak) | 1 (fail) | Schema codes |
|---|---|---|---|---|
| **Grammaticality** | Stem and options are fluent and read only one way | Awkward wording, but the meaning is clear | Ungrammatical or ambiguous: *"Why start the garden students did?"* | I1 (stem); D3 (option doesn't fit stem) |
| **Answerability** | Exactly one option is supported, and it is the key | The key is right but the support is indirect, so the annotator hesitates (use notes) | Key wrong or unsupported, two options defensible, or the fact isn't in the passage | I2, I3 (via D1), I5; `matches_key` is direct evidence |
| **Relevance** | Asks about a main idea or important detail of the passage | Asks about a peripheral detail (e.g. a date in passing) | Off-passage | I5 for score 1 only; 2 vs. 3 is **new** |
| **Non-triviality** | Needs reading, often across sentences: *"Why did the students start the garden?"* when the reason is implied | Lifts the answer from one sentence | Guessable without the passage (only one option echoes the stem) | I4; D2/D4 weak distractors are a common cause |

**What the rubric adds:** a graded relevance judgment. The schema deferred "central concept" until the generation pipeline existed, and it now does. The rubric also splits "lookup" from "inference" in non-triviality. This touches cognitive level, which the schema left out of scope, so treat the 2/3 split as exploratory. **What the schema covers but the rubric doesn't:** per-distractor labels D1–D4 and their roll-up rules, annotator time, and agreement procedures. Keep using the schema for those. Annotators record schema labels first, then rubric scores.

## 3. What Week 5 should compute

Compute **ROUGE-L and BERTScore F1** (best match over gold stems) as cheap similarity signals, plus **BLEU-4** only so results can be compared with published QG numbers [S]. Add one copy check, **ROUGE-L of the generated stem against the passage itself**, to flag questions that just lift sentences. Skip METEOR, which adds little over ROUGE [9], and perplexity, which measures only fluency and depends on the scoring LM. Neither tells us whether an MCQ works. If budget allows, add the **QA-based answerability check** (a model answers each item, then we compare its answer to the key). It is the only automatic signal that touches correctness, which [8] and [9] show overlap metrics miss. Final quality claims should rest on the human rubric and schema labels, with automatic scores reported as exploratory counts and means over 20 passages.

---
**Sources.** [8] Zhang et al., *BERTScore*, ICLR 2020, arXiv:1904.09675. [9] Scialom et al., *QuestEval*, EMNLP 2021, arXiv:2103.12693. [S] Kurdi et al., *A Systematic Review of AQG for Educational Purposes*, IJAIED 2020 (`Papers/AQG.pdf`, Table 9 and "Quality Criteria and Metrics"). [7] C. Dilmegani & S. Ermut, *10+ Large Language Model Examples*, AIMultiple, updated 2026-10-09, https://aimultiple.com/large-language-models-examples. It compares models only by public benchmarks and latency and says nothing about BLEU/ROUGE/METEOR/BERTScore, perplexity, human evaluation or LLM-as-judge, so no claim above relies on it.
