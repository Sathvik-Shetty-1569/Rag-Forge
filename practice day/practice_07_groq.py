from groq import Groq
from dotenv import load_dotenv
import os


load_dotenv()
client = Groq()

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role":"user","content": "What is Retrieval-Augmented Generation?"}]   
)

fake_retrieved_chunks = [
    "RAG models achieve state-of-the-art results on Natural Questions, WebQuestions and CuratedTrec, outperforming parametric seq2seq models.",
    "RAG combines parametric memory (the LLM) with non-parametric memory (a dense vector index of Wikipedia accessed with a neural retriever).",
    "For language generation tasks, RAG models generate more specific, diverse and factual language than a state-of-the-art parametric-only seq2seq baseline."
]

# Format chunks into a single context block
context = "\n\n".join([f"[chunk {i+1}]: {chunk}" 
                        for i, chunk in enumerate(fake_retrieved_chunks)])

# This is the standard RAG prompt structure
system_prompt = f"""You are a helpful research assistant. 
Answer the user's question using ONLY the context provided below.
If the context doesn't contain enough information to answer, say 'I don't have enough context to answer this.'
Do not use any outside knowledge.

Context:
{context}
"""

response2 = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What are the benefits of RAG over parametric-only models?"}
    ]
)

print("\n=== RAG-style prompt (with context) ===")
print(response2.choices[0].message.content)