# QuestGen

**Research Topic:** Using different question generation methods to test whether prompting can affect how well LLM-generated questions test passage comprehension.

## Week 1

- Selected target audience: Grades 9th–12th
- Selected source dataset: RACE
- Cleaned RACE to remove duplicate passages, questions referencing underlined passages, and thin passages (fewer than 2 questions). Used Flesch-Kincaid to assign a grade level to each passage and filtered the dataset to grades 9–12.

**Corpus Details:** 7,042 passages (rows)

**To build the corpus, run:**

pip install datasets
generate_seed_corpus.py


**Future Work:**
Define a question schema (stem, answer, distractors, question type, cognitive level, difficulty, source span) and an annotation guideline document. State explicitly what "difficulty" denotes in the schema.
