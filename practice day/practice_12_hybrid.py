"""
practice_12_hybrid.py
Goal: combine BM25 + dense retrieval using Reciprocal Rank Fusion (RRF).
RRF doesn't care about raw scores — only about RANK POSITION.
"""

import re
import numpy as np
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


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


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_dense(query, chunks, embeddings, embed_model, k=10):
    """Return top-k (score, idx) pairs using dense retrieval."""
    query_emb = embed_model.encode(query)
    scores = [(cosine_similarity(query_emb, emb), i)
              for i, emb in enumerate(embeddings)]
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[:k]


def retrieve_bm25(query, chunks, bm25, k=10):
    """Return top-k (score, idx) pairs using BM25."""
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(score, idx) for idx, score in ranked[:k]]


def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    """
    RRF merges rankings from multiple retrievers.

    Key insight: raw scores from BM25 and dense are NOT comparable.
    BM25 scores are 0-15, dense scores are 0-1 — you can't average them.

    RRF ignores raw scores entirely. It only cares about RANK POSITION.
    Formula for each chunk: sum of 1/(k + rank) across all retrievers.

    k=60 is a constant that dampens the influence of very high ranks.
    A chunk ranked #1 by both retrievers scores highest.
    A chunk ranked #1 by one and missing from the other still scores well.
    """
    rrf_scores = {}

    # Dense results: (score, idx) pairs already sorted by rank
    for rank, (score, idx) in enumerate(dense_results, 1):
        if idx not in rrf_scores:
            rrf_scores[idx] = 0
        rrf_scores[idx] += 1 / (k + rank)

    # BM25 results: (score, idx) pairs already sorted by rank
    for rank, (score, idx) in enumerate(bm25_results, 1):
        if idx not in rrf_scores:
            rrf_scores[idx] = 0
        rrf_scores[idx] += 1 / (k + rank)

    # Sort by RRF score descending
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


if __name__ == "__main__":
    pdf_path = Path("data/papers/2005.11401v4.pdf")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading, chunking, embedding...")
    text = load_and_clean_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = embed_model.encode(chunks)

    print("Building BM25 index...")
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    print(f"{len(chunks)} chunks ready\n")

    # Test on BOTH queries from practice_11
    queries = [
        "How does RAG reduce hallucination?",
        "TriviaQA Natural Questions WebQuestions results",
    ]

    for query in queries:
        print("=" * 60)
        print(f"Query: {query}")
        print("=" * 60)

        # Get top-10 from each retriever (wider net before fusion)
        dense_results = retrieve_dense(query, chunks, embeddings, embed_model, k=10)
        bm25_results = retrieve_bm25(query, chunks, bm25, k=10)

        # Fuse rankings
        fused = reciprocal_rank_fusion(dense_results, bm25_results)

        print("\n--- HYBRID (RRF) top 3 ---")
        for rank, (idx, rrf_score) in enumerate(fused[:3], 1):
            print(f"\nRank {rank} (RRF score: {rrf_score:.4f})")
            print(chunks[idx][:200])
        print()