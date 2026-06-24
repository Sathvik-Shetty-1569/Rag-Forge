from pathlib import Path
from pypdf import PdfReader
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

def textcleaner(file:Path)-> str:
    reader = PdfReader(str(file))
    full_text = []
    for pages in reader.pages:
        text = pages.extract_text()
        if text:
            full_text.append(text)
    raw = "\n".join(full_text)
    cleaned = re.sub(r"[ \t\n]+", " ", raw)
    cleaned = cleaned.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return cleaned

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[". ", " ", ""],  # no \n\n since we already collapsed newlines
    )
    return splitter.split_text(text)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_top_k(query: str, chunks: list[str], chunk_embeddings, model, k: int = 3):
    """Embed the query, compare against every chunk, return the best k."""
    query_embedding = model.encode(query)
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3,
    )

    return [(results['distances'][0][i], results['documents'][0][i]) for i in range(0,3)]


if __name__ == "__main__":
    pdf_path = Path("data/papers/2005.11401v4.pdf")

    print("Loading and cleaning PDF...")
    text = textcleaner(pdf_path)

    print("Chunking...")
    chunks = chunk_text(text)
    print(f"Got {len(chunks)} chunks\n")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding all chunks (this may take a moment)...")
    chunk_embeddings = model.encode(chunks)

    client = chromadb.PersistentClient(path="./chroma")
    collection = client.get_or_create_collection(name='my_docs')
    collection.upsert(
        ids=[f"doc_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=chunk_embeddings
    )
    # Real question about RAG
    query = "RAG models generate more factual and specific language"

    print(f"\nQuery: {query}\n")
    results = retrieve_top_k(query, chunks, chunk_embeddings, model, k=3)

    for rank, (score, chunk) in enumerate(results, start=1):
        print(f"--- Rank {rank} (score: {score:.4f}) ---")
        print(chunk)
        print()