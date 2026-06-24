"""
practice_16_golden_dataset.py
Goal: build a small golden Q&A dataset from our RAG paper.
These are hand-crafted question/answer pairs where WE know the correct answer.
RAGAS will compare our pipeline's output against these ground truths.
"""

import json
from pathlib import Path

# Hand-crafted golden dataset — 8 questions covering different parts of the paper.
# ground_truth: the correct answer according to the paper
# relevant_chunk: a snippet that contains the answer (for context recall scoring)

golden_dataset = [
    {
        "question": "What datasets were used to evaluate RAG models?",
        "ground_truth": "RAG models were evaluated on Natural Questions, WebQuestions, CuratedTrec, TriviaQA, MS MARCO, Jeopardy question generation, and FEVER fact verification.",
        "relevant_chunk": "Our RAG models achieve state-of-the-art results on open Natural Questions, WebQuestions and CuratedTrec and strongly outperform recent approaches on TriviaQA."
    },
    {
        "question": "How does RAG combine parametric and non-parametric memory?",
        "ground_truth": "RAG uses a pre-trained seq2seq model as parametric memory and a dense vector index of Wikipedia accessed with a neural retriever as non-parametric memory.",
        "relevant_chunk": "We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever."
    },
    {
        "question": "What are the two RAG formulations introduced in the paper?",
        "ground_truth": "RAG-Sequence conditions on the same retrieved passages across the whole generated sequence, while RAG-Token can use different passages per token.",
        "relevant_chunk": "We compare two RAG formulations, one which conditions on the same retrieved passages across the whole generated sequence, and another which can use different passages per token."
    },
    {
        "question": "What retriever does RAG use?",
        "ground_truth": "RAG uses a Dense Passage Retriever (DPR) as its retrieval component, initialized using DPR's retriever which uses retrieval supervision on Natural Questions and TriviaQA.",
        "relevant_chunk": "RAG's retriever is initialized using DPR's retriever, which uses retrieval supervision on Natural Questions and TriviaQA."
    },
    {
        "question": "How does RAG handle knowledge that changes over time?",
        "ground_truth": "RAG can be updated by replacing the non-parametric memory index without retraining the model, making it easy to update knowledge.",
        "relevant_chunk": "RAG's non-parametric memory can be replaced and updated without retraining the model, allowing knowledge to be updated efficiently."
    },
    {
        "question": "What metric was used for Jeopardy question generation evaluation?",
        "ground_truth": "The SQuAD-tuned Q-BLEU-1 metric was used for Jeopardy question generation evaluation.",
        "relevant_chunk": "Following prior work, we evaluate using the SQuAD-tuned Q-BLEU-1 metric, a variant of BLEU with a higher weight for matching entities."
    },
    {
        "question": "What does RAG-Token allow that RAG-Sequence does not?",
        "ground_truth": "RAG-Token allows using different retrieved passages for each generated token, while RAG-Sequence uses the same passages for the entire sequence.",
        "relevant_chunk": "RAG-Token can use different passages per token during generation, unlike RAG-Sequence which conditions on the same passages for the whole sequence."
    },
    {
        "question": "How does RAG compare to closed book QA approaches?",
        "ground_truth": "RAG outperforms closed book QA approaches by combining retrieval with generation, achieving better accuracy on open domain QA tasks without relying purely on parametric knowledge.",
        "relevant_chunk": "We compare to Closed-Book QA approaches which generate answers but do not exploit retrieval, instead relying purely on parametric knowledge."
    },
]

# Save to disk so practice_17 can load it
output_path = Path("data/golden_dataset.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    json.dump(golden_dataset, f, indent=2)

print(f"Golden dataset saved: {len(golden_dataset)} Q&A pairs")
print(f"Location: {output_path}\n")

print("Sample entry:")
print(f"  Question: {golden_dataset[0]['question']}")
print(f"  Answer:   {golden_dataset[0]['ground_truth'][:80]}...")