from langchain_text_splitters import CharacterTextSplitter , RecursiveCharacterTextSplitter

sample_text = """Retrieval-Augmented Generation combines two ideas. First, a retriever finds relevant documents from a large corpus. Second, a generator uses those documents to produce an answer.

This is powerful because language models often hallucinate facts. By grounding the generation step in real retrieved text, RAG reduces hallucination significantly.

However, retrieval quality matters a lot. If the retriever pulls irrelevant chunks, the generator will produce a wrong or misleading answer, no matter how good the language model is."""

print("ORIGINAL TEXT LENGTH:", len(sample_text), "characters\n")

fixed_splitter = CharacterTextSplitter(
    separator="",
    chunk_size = 150,
    chunk_overlap = 20
)

fixed_chunks = fixed_splitter.split_text(sample_text)

print("=" * 60)
print(f"FIXED-SIZE CHUNKING -> {len(fixed_chunks)} chunks")
print("=" * 60)
for i, chunk in enumerate(fixed_chunks):
    print(f"\n[Chunk {i}] ({len(chunk)} chars)")
    print(chunk)
    
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=40,
    chunk_overlap=20,
    separators=["\n\n", ". ", " ", ""],
)
recursive_chunks = recursive_splitter.split_text(sample_text)

print("\n\n" + "=" * 60)
print(f"RECURSIVE CHUNKING -> {len(recursive_chunks)} chunks")
print("=" * 60)
for i, chunk in enumerate(recursive_chunks):
    print(f"\n[Chunk {i}] ({len(chunk)} chars)")
    print(chunk)