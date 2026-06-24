"""
practice_14_parent_retriever.py
Goal: implement parent-document retrieval.
Small child chunks go into Chroma for retrieval.
Large parent chunks stay in memory for generation.
At query time: retrieve child → look up parent → pass parent to LLM.
"""

import re
import uuid
import numpy as np
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb


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


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_parent_child_chunks(text: str):
    """
    Split text into large parent chunks first.
    Then split each parent into small child chunks.
    Each child carries its parent's ID so we can look it up later.
    """
    # Large chunks for generation context
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        separators=[". ", " ", ""],
    )

    # Small chunks for retrieval (tight, focused embeddings)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        separators=[". ", " ", ""],
    )

    parent_chunks = parent_splitter.split_text(text)

    # parent_store: {parent_id: parent_text}
    parent_store = {}

    # child_chunks: list of {text, parent_id}
    child_chunks = []

    for parent_text in parent_chunks:
        parent_id = str(uuid.uuid4())  # unique ID for this parent
        parent_store[parent_id] = parent_text

        children = child_splitter.split_text(parent_text)
        for child_text in children:
            child_chunks.append({
                "text": child_text,
                "parent_id": parent_id
            })

    return parent_store, child_chunks


def retrieve_with_parents(query: str, collection, parent_store: dict,
                          embed_model, k: int = 3):
    """
    Query Chroma with the query → get matching child chunks.
    Look up each child's parent_id → return parent chunks to LLM.
    Deduplicate: multiple children from same parent → return parent once.
    """
    query_emb = embed_model.encode([query])

    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=k,
        include=["documents", "metadatas"]
    )

    # Extract parent_ids from matched children's metadata
    matched_metadatas = results["metadatas"][0]
    matched_children = results["documents"][0]

    seen_parent_ids = set()
    parent_results = []

    for child_text, metadata in zip(matched_children, matched_metadatas):
        parent_id = metadata["parent_id"]

        print(f"\nChild matched: {child_text[:100]}...")
        print(f"Parent ID: {parent_id[:8]}...")

        if parent_id not in seen_parent_ids:
            seen_parent_ids.add(parent_id)
            parent_text = parent_store[parent_id]
            parent_results.append(parent_text)

    return parent_results


if __name__ == "__main__":
    pdf_path = Path("data/papers/2005.11401v4.pdf")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading and cleaning PDF...")
    text = load_and_clean_pdf(pdf_path)

    print("Building parent-child chunks...")
    parent_store, child_chunks = build_parent_child_chunks(text)

    print(f"Parents: {len(parent_store)}")
    print(f"Children: {len(child_chunks)}")
    print(f"Average children per parent: {len(child_chunks)/len(parent_store):.1f}\n")

    # Store children in Chroma with parent_id as metadata
    print("Embedding children and storing in Chroma...")
    child_texts = [c["text"] for c in child_chunks]
    child_embeddings = embed_model.encode(child_texts)

    client = chromadb.PersistentClient(path="./chroma_parent")
    collection = client.get_or_create_collection(
        name="parent_child",
        metadata={"hnsw:space": "cosine"}
    )
    collection.upsert(
        ids=[f"child_{i}" for i in range(len(child_chunks))],
        documents=child_texts,
        embeddings=child_embeddings.tolist(),
        metadatas=[{"parent_id": c["parent_id"]} for c in child_chunks]
    )

    query = "How does RAG reduce hallucination?"
    print(f"\nQuery: {query}")
    print("=" * 60)

    parent_results = retrieve_with_parents(
        query, collection, parent_store, embed_model, k=3
    )

    print("\n--- PARENT CHUNKS RETURNED TO LLM ---")
    for rank, parent_text in enumerate(parent_results, 1):
        print(f"\nParent {rank} ({len(parent_text)} chars):")
        print(parent_text[:300])
        print("...")