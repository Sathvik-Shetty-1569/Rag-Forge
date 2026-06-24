"""
practice_13_reranking.py
Goal: understand the difference between bi-encoder (what we've been using)
and cross-encoder (reranker), and see reranking improve our top-3.

Bi-encoder:   embed query separately, embed chunk separately, compare vectors.
              Fast — embeddings are precomputed. But query and chunk never
              "see" each other during encoding, so nuance is lost.

Cross-encoder: takes (query, chunk) TOGETHER as one input, outputs a single
               relevance score. Much more accurate because the model can attend
               to interactions between query words and chunk words directly.
               Slow — must run for every (query, chunk) pair at query time.
"""

import re
import numpy as np
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
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


def retrieve_candidates(query, chunks, embeddings, embed_model, bm25, k=10):
    """Get top-k candidates from both retrievers, merge with RRF.
    This is our wide net — we'll rerank these candidates next."""
    # Dense
    query_emb = embed_model.encode(query)
    dense_scores = [(cosine_similarity(query_emb, emb), i)
                    for i, emb in enumerate(embeddings)]
    dense_scores.sort(reverse=True, key=lambda x: x[0])
    dense_top = dense_scores[:k]

    # BM25
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
    bm25_top = [(score, idx) for idx, score in bm25_ranked[:k]]

    # RRF fusion
    rrf_scores = {}
    for rank, (_, idx) in enumerate(dense_top, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank)
    for rank, (_, idx) in enumerate(bm25_top, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank)

    sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Return top-k candidate chunks
    return [chunks[idx] for idx, _ in sorted_candidates[:k]]


def rerank(query: str, candidate_chunks: list[str], cross_encoder: CrossEncoder, k: int = 3):
    """
    Cross-encoder scores each (query, chunk) pair together.
    It reads both at once — can catch word interactions bi-encoder misses.
    Returns top-k reranked chunks.
    """
    # Build pairs: [(query, chunk1), (query, chunk2), ...]
    pairs = [(query, chunk) for chunk in candidate_chunks]

    # Score all pairs — this is the slow step (runs inference for each pair)
    scores = cross_encoder.predict(pairs)

    # Zip scores with chunks and sort
    scored_chunks = sorted(zip(scores, candidate_chunks), reverse=True)
    return scored_chunks[:k]


if __name__ == "__main__":
    pdf_path = Path("data/papers/2005.11401v4.pdf")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("Loading, chunking, embedding...")
    text = load_and_clean_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = embed_model.encode(chunks)

    print("Building BM25 index...")
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    print(f"{len(chunks)} chunks ready\n")

    query = "How does RAG reduce hallucination?"

    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)

    # Step 1 — retrieve top-10 candidates via hybrid
    candidates = retrieve_candidates(
        query, chunks, embeddings, embed_model, bm25, k=10
    )
    print(f"\nCandidates from hybrid retrieval: {len(candidates)}")

    # Step 2 — rerank candidates with cross-encoder
    print("Reranking with cross-encoder...")
    reranked = rerank(query, candidates, cross_encoder, k=3)

    print("\n--- AFTER RERANKING top 3 ---")
    for rank, (score, chunk) in enumerate(reranked, 1):
        print(f"\nRank {rank} (cross-encoder score: {score:.4f})")
        print(chunk[:200])