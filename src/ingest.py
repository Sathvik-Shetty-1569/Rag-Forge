"""
ingest.py — loads PDFs, chunks them with multiple strategies, and lets us
compare chunk quality before committing to one approach.
"""

from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

PAPERS_DIR = Path("data/papers")


def load_pdf_text(filepath: Path) -> str:
    """Extract raw text from a PDF, page by page."""
    reader = PdfReader(str(filepath))
    full_text = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text.append(text)
    return "\n".join(full_text)


def load_all_papers() -> dict[str, str]:
    """Load every PDF in data/papers/ into {filename: raw_text}."""
    documents = {}
    for pdf_file in PAPERS_DIR.glob("*.pdf"):
        print(f"Loading {pdf_file.name}...")
        documents[pdf_file.name] = load_pdf_text(pdf_file)
    return documents


def chunk_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Naive fixed-size character chunking. Fast but ignores structure —
    will cut mid-sentence, mid-equation, mid-table."""
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_text(text)


def chunk_recursive(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Recursive splitting — tries paragraph breaks first, then sentences,
    then words. Respects document structure much better than fixed-size."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


if __name__ == "__main__":
    docs = load_all_papers()

    # Compare both strategies on the first paper
    sample_name = list(docs.keys())[0]
    sample_text = docs[sample_name]

    print(f"\n--- Comparing chunking strategies on: {sample_name} ---")
    print(f"Total chars: {len(sample_text)}")

    fixed_chunks = chunk_fixed_size(sample_text)
    recursive_chunks = chunk_recursive(sample_text)

    print(f"\nFixed-size chunking: {len(fixed_chunks)} chunks")
    print(f"Recursive chunking: {len(recursive_chunks)} chunks")

    print("\n--- Sample fixed-size chunk (#5) ---")
    print(fixed_chunks[5] if len(fixed_chunks) > 5 else fixed_chunks[0])

    print("\n--- Sample recursive chunk (#5) ---")
    print(recursive_chunks[5] if len(recursive_chunks) > 5 else recursive_chunks[0])