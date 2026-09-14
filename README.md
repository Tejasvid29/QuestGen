Project Name: QuestGen

Research Topic: Using different Question Generation methods to test whether
prompting can affect how well LLM generated questions test passage comprehension. 

Week 1: 
Selected target audience: Grade 9th - 12th
Selected Source Dataset: RACE 
Cleaned RACE to remove duplication, questions referecing underlined passages, removed thin passages (less than 2 questions), used Flesch-Kincaid
to assign grade level to passages and filtered grade 9-12 from the dataset. 

Corpus Details: 7042 passages (rows)

To build the corpus run:
pip install datasets
generate_seed_corpus.py

Future work:
Define a question schema (stem, answer, distractors, question type, cognitive level, difficulty, source span) and an annotation guideline document. State explicitly what "difficulty" denotes in the schema.