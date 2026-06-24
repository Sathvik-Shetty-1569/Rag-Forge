"""
practice_09_hyde.py
Goal: understand HyDE — Hypothetical Document Embeddings.
Instead of embedding the raw question, we generate a hypothetical answer
and embed that, since it lives closer to real document text in vector space.
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


# ── Helpers (same as before) ──────────────────────────────────────

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
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[". ", " ", ""],
    )
    return splitter.split_text(text)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_naive(query: str, chunks: list[str], embeddings, model, k=3):
    """Standard retrieval — embed the raw question."""
    query_emb = model.encode(query)
    scores = [(cosine_similarity(query_emb, emb), i)
              for i, emb in enumerate(embeddings)]
    scores.sort(reverse=True, key=lambda x: x[0])
    return [(score, chunks[i]) for score, i in scores[:k]]


# ── HyDE ─────────────────────────────────────────────────────────

def generate_hypothetical_answer(query: str, groq_client: Groq) -> str:
    """Ask the LLM to write a hypothetical answer as if it came from a paper.
    We don't care if it's factually correct — we just need it to LOOK like
    the kind of text that would appear in a research paper about this topic."""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research paper writer. "
                    "Write a short 2-3 sentence passage that directly answers "
                    "the following question, as if it were extracted from an "
                    "academic NLP paper. Be specific and use academic language."
                )
            },
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content


def retrieve_hyde(query: str, chunks: list[str], embeddings, 
                  embed_model, groq_client: Groq, k=3):
    """HyDE retrieval — embed hypothetical answer instead of raw question."""
    hypothetical_answer = generate_hypothetical_answer(query, groq_client)
    print(f"\nHypothetical answer generated:\n{hypothetical_answer}\n")

    # Embed the hypothetical answer, not the original question
    hypo_emb = embed_model.encode(hypothetical_answer)
    scores = [(cosine_similarity(hypo_emb, emb), i)
              for i, emb in enumerate(embeddings)]
    scores.sort(reverse=True, key=lambda x: x[0])
    return [(score, chunks[i]) for score, i in scores[:k]]


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

    # The weak query we proved was problematic on Day 2
    query = "How does RAG reduce hallucination?"

    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)

    # Naive retrieval
    print("\n--- NAIVE RETRIEVAL (embed raw question) ---")
    naive_results = retrieve_naive(query, chunks, embeddings, embed_model, k=3)
    for rank, (score, chunk) in enumerate(naive_results, 1):
        print(f"\nRank {rank} (score: {score:.4f})")
        print(chunk[:200])  # first 200 chars so output stays readable

    # HyDE retrieval
    print("\n--- HYDE RETRIEVAL (embed hypothetical answer) ---")
    hyde_results = retrieve_hyde(
        query, chunks, embeddings, embed_model, groq_client, k=3
    )
    for rank, (score, chunk) in enumerate(hyde_results, 1):
        print(f"\nRank {rank} (score: {score:.4f})")
        print(chunk[:200])