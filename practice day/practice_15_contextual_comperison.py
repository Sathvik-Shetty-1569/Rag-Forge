"""
practice_15_contextual_compression.py
Goal: after retrieving large parent chunks, use an LLM to compress each
chunk down to only the sentences relevant to the query.

Without compression: pass 1900 chars to LLM (contains relevant + irrelevant)
With compression:    pass ~300 chars to LLM (only the relevant part)
"""

import re
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import chromadb
import uuid

load_dotenv()


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


def build_parent_child_chunks(text: str):
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        separators=[". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        separators=[". ", " ", ""],
    )
    parent_chunks = parent_splitter.split_text(text)
    parent_store = {}
    child_chunks = []

    for parent_text in parent_chunks:
        parent_id = str(uuid.uuid4())
        parent_store[parent_id] = parent_text
        for child_text in child_splitter.split_text(parent_text):
            child_chunks.append({"text": child_text, "parent_id": parent_id})

    return parent_store, child_chunks


def retrieve_parents(query, collection, parent_store, embed_model, k=3):
    query_emb = embed_model.encode([query])
    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=k,
        include=["documents", "metadatas"]
    )
    seen = set()
    parents = []
    for metadata in results["metadatas"][0]:
        pid = metadata["parent_id"]
        if pid not in seen:
            seen.add(pid)
            parents.append(parent_store[pid])
    return parents


def compress_chunk(query: str, chunk: str, groq_client: Groq) -> str:
    """
    Ask the LLM to extract only the sentences from the chunk
    that are relevant to the query. If nothing is relevant,
    return empty string so we can skip this chunk entirely.
    """
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a context compression assistant. "
                    "Given a query and a passage, extract ONLY the sentences "
                    "from the passage that are directly relevant to answering the query. "
                    "Return the extracted sentences verbatim with no modifications. "
                    "If no sentences are relevant, return exactly: NO_RELEVANT_CONTENT"
                )
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nPassage:\n{chunk}"
            }
        ]
    )
    result = response.choices[0].message.content.strip()
    if result == "NO_RELEVANT_CONTENT":
        return ""
    return result


if __name__ == "__main__":
    pdf_path = Path("data/papers/2005.11401v4.pdf")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    groq_client = Groq()

    print("Loading and building chunks...")
    text = load_and_clean_pdf(pdf_path)
    parent_store, child_chunks = build_parent_child_chunks(text)

    child_texts = [c["text"] for c in child_chunks]
    child_embeddings = embed_model.encode(child_texts)

    client = chromadb.PersistentClient(path="./chroma_compression")
    collection = client.get_or_create_collection(
        name="compression_test",
        metadata={"hnsw:space": "cosine"}
    )
    collection.upsert(
        ids=[f"child_{i}" for i in range(len(child_chunks))],
        documents=child_texts,
        embeddings=child_embeddings.tolist(),
        metadatas=[{"parent_id": c["parent_id"]} for c in child_chunks]
    )

    query = "How does RAG reduce hallucination?"
    print(f"\nQuery: {query}\n")

    # Step 1 — retrieve parent chunks
    parents = retrieve_parents(query, collection, parent_store, embed_model, k=3)

    # Step 2 — compress each parent
    print("Compressing chunks...\n")
    compressed = []
    for i, parent in enumerate(parents, 1):
        print(f"Parent {i} — original: {len(parent)} chars")
        result = compress_chunk(query, parent, groq_client)

        if result:
            print(f"Parent {i} — compressed: {len(result)} chars")
            print(f"Compressed content: {result[:200]}")
            compressed.append(result)
        else:
            print(f"Parent {i} — no relevant content, skipping")
        print()

    print("=" * 60)
    print(f"Final context passed to LLM: {len(compressed)} chunks")
    total_chars = sum(len(c) for c in compressed)
    print(f"Total chars: {total_chars} (vs ~{len(parents) * 1900} without compression)")