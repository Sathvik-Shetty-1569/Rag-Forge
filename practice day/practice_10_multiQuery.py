"""
practice_10_multiquery.py
Goal: instead of one query that might miss, generate 3-4 rephrasings,
retrieve for each, merge results, deduplicate.
If one phrasing fails, others compensate.
"""

import re
import numpy as np
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


# ── Helpers ───────────────────────────────────────────────────────

def load_and_clean_pdf(filepath: Path) -> str:
    reader = PdfReader(str(filepath))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    raw = "\n".join(pages)
    cleaned = re.sub(r"[ \t\n]+", " ", raw)
    cleaned = cleaned.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return cleaned


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[". ", " ", ""],
    )
    return splitter.split_text(text)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ── Multi-query ───────────────────────────────────────────────────

def generate_rephrasings(query: str, groq_client: Groq) -> list[str]:
    """Ask the LLM to rephrase the query 3 different ways.
    Each rephrasing might use different vocabulary, hitting different chunks."""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate exactly 3 different rephrasings of the user's question. "
                    "Each rephrasing should use different vocabulary and sentence structure "
                    "but preserve the original meaning. "
                    "Return ONLY the 3 questions, one per line, no numbering, no extra text."
                )
            },
            {"role": "user", "content": query}
        ]
    )
    raw = response.choices[0].message.content
    # Split into individual questions, drop empty lines
    rephrasings = [line.strip() for line in raw.split("\n") if line.strip()]
    return rephrasings[:3]  # safety — take max 3


def retrieve_multiquery(query: str, chunks: list[str], embeddings,
                        embed_model, groq_client: Groq, k: int = 3) -> list[tuple]:
    """Retrieve for original query + 3 rephrasings, merge, deduplicate."""
    # All queries: original + rephrasings
    rephrasings = generate_rephrasings(query, groq_client)
    all_queries = [query] + rephrasings

    print("\nAll queries used for retrieval:")
    for i, q in enumerate(all_queries):
        label = "ORIGINAL" if i == 0 else f"Rephrasing {i}"
        print(f"  [{label}]: {q}")

    # Retrieve top-k for each query, collect all (score, chunk_index) pairs
    all_scores = []
    
    for q in all_queries:
        q_emb = embed_model.encode(q)
        scores = [(cosine_similarity(q_emb, emb), i)
                  for i, emb in enumerate(embeddings)]
        all_scores.extend(scores)  # dump everything in
    
    # Sort entire pool by score, highest first
    all_scores.sort(reverse=True, key=lambda x: x[0])
    
    # Walk through, pick first (highest) occurrence of each idx
    seen = set()
    merged_results = []
    
    for score, idx in all_scores:
        if idx not in seen:
            seen.add(idx)
            merged_results.append((score, chunks[idx]))
        if len(merged_results) == k:
            break
    return merged_results


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    pdf_path = Path("data/papers/2005.11401v4.pdf")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    groq_client = Groq()

    print("Loading, chunking, embedding...")
    text = load_and_clean_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = embed_model.encode(chunks)
    print(f"{len(chunks)} chunks ready\n")

    query = "How does RAG reduce hallucination?"

    print("=" * 60)
    print(f"Original query: {query}")
    print("=" * 60)

    results = retrieve_multiquery(
        query, chunks, embeddings, embed_model, groq_client, k=3
    )

    print("\n--- MULTI-QUERY RESULTS ---")
    for rank, (score, chunk) in enumerate(results, 1):
        print(f"\nRank {rank} (score: {score:.4f})")
        print(chunk[:200])