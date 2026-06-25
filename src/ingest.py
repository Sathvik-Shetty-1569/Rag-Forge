"""
ingest.py — RAG-Forge ingestion pipeline
Loads PDFs, cleans text, chunks, embeds, stores in Chroma.
Supports multiple papers in one collection.
"""

import re
import uuid
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb


PAPERS_DIR    = Path("data/papers")
CHROMA_PATH   = Path("./chroma_forge")
COLLECTION    = "rag_forge"
EMBED_MODEL   = "all-MiniLM-L6-v2"

# Chunk sizes — small children for retrieval, large parents for generation
PARENT_SIZE   = 2000
PARENT_OVERLAP = 100
CHILD_SIZE    = 200
CHILD_OVERLAP  = 20


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


def is_garbage_chunk(chunk: str) -> bool:
    """Filter bibliography, headers, and other low-content chunks."""
    signals = 0
    if "http://" in chunk or "https://" in chunk:
        signals += 1
    citations = re.findall(r'\[\d+\]', chunk)
    if len(citations) > 6:
        signals += 1
    if len(chunk.strip()) < 100:
        signals += 1
    return signals >= 2


def build_parent_child_chunks(text: str):
    """Split into large parents for generation, small children for retrieval."""
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_SIZE,
        chunk_overlap=PARENT_OVERLAP,
        separators=[". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_SIZE,
        chunk_overlap=CHILD_OVERLAP,
        separators=[". ", " ", ""],
    )

    parent_store = {}
    child_chunks  = []

    for parent_text in parent_splitter.split_text(text):
        if is_garbage_chunk(parent_text):
            continue
        parent_id = str(uuid.uuid4())
        parent_store[parent_id] = parent_text
        for child_text in child_splitter.split_text(parent_text):
            if not is_garbage_chunk(child_text):
                child_chunks.append({
                    "text": child_text,
                    "parent_id": parent_id,
                })

    return parent_store, child_chunks


def get_collection():
    """Return the Chroma collection (create if not exists)."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_papers(papers_dir: Path = PAPERS_DIR):
    """
    Main ingestion function.
    Loads every PDF in papers_dir, builds parent-child chunks,
    embeds children, stores in Chroma.
    Returns parent_store so pipeline can look up full context.
    """
    embed_model = SentenceTransformer(EMBED_MODEL)
    collection  = get_collection()

    all_parent_store = {}
    all_child_chunks = []

    pdf_files = list(papers_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} papers\n")

    for pdf_path in pdf_files:
        print(f"Ingesting: {pdf_path.name}")
        text = load_and_clean_pdf(pdf_path)
        parent_store, child_chunks = build_parent_child_chunks(text)

        print(f"  Parents: {len(parent_store)} | Children: {len(child_chunks)}")

        all_parent_store.update(parent_store)
        all_child_chunks.extend(child_chunks)

    # Embed all children at once (faster than per-paper)
    print(f"\nEmbedding {len(all_child_chunks)} child chunks...")
    child_texts = [c["text"] for c in all_child_chunks]
    embeddings  = embed_model.encode(child_texts, show_progress_bar=True)

    # Upsert into Chroma
    print("Storing in Chroma...")
    collection.upsert(
        ids        = [f"child_{i}" for i in range(len(all_child_chunks))],
        documents  = child_texts,
        embeddings = embeddings.tolist(),
        metadatas  = [{"parent_id": c["parent_id"]} for c in all_child_chunks],
    )

    print(f"\nIngestion complete.")
    print(f"Total parents (full context chunks): {len(all_parent_store)}")
    print(f"Total children (retrieval chunks):   {len(all_child_chunks)}")

    return all_parent_store


if __name__ == "__main__":
    parent_store = ingest_papers()
    print("\nSample parent chunk (first 200 chars):")
    sample = list(parent_store.values())[0]
    print(sample[:200])