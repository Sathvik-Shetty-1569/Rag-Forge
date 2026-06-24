"""
practice_11_bm25.py
Goal: understand BM25 retrieval on its own before combining with dense.
See what it gets right that dense missed, and where it fails.
"""

import re
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    """BM25 works on token lists, not raw strings.
    Simple approach: lowercase, split on whitespace.
    In production you'd also remove stopwords and stem."""
    return text.lower().split()


def retrieve_bm25(query: str, chunks: list[str], bm25: BM25Okapi, k: int = 3):
    """Score all chunks against query using BM25, return top k."""
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Pair scores with chunk indices, sort descending
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(score, chunks[idx]) for idx, score in ranked[:k]]


if __name__ == "__main__":
    pdf_path = Path("data/papers/2005.11401v4.pdf")

    print("Loading and chunking...")
    text = load_and_clean_pdf(pdf_path)
    chunks = chunk_text(text)
    print(f"{len(chunks)} chunks\n")

    # Build BM25 index from tokenized chunks
    print("Building BM25 index...")
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    # Test 1 — semantic query (BM25 should struggle here)
    query1 = "How does RAG reduce hallucination?"
    print("=" * 60)
    print(f"Query 1 (semantic): {query1}")
    print("=" * 60)
    results1 = retrieve_bm25(query1, chunks, bm25, k=3)
    for rank, (score, chunk) in enumerate(results1, 1):
        print(f"\nRank {rank} (BM25 score: {score:.4f})")
        print(chunk[:200])

    # Test 2 — exact keyword query (BM25 should shine here)
    query2 = "TriviaQA Natural Questions WebQuestions results"
    print("\n" + "=" * 60)
    print(f"Query 2 (exact keywords): {query2}")
    print("=" * 60)
    results2 = retrieve_bm25(query2, chunks, bm25, k=3)
    for rank, (score, chunk) in enumerate(results2, 1):
        print(f"\nRank {rank} (BM25 score: {score:.4f})")
        print(chunk[:200])