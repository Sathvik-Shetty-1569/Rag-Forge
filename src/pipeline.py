"""
pipeline.py — RAG-Forge main pipeline
Wires ingest → retriever → generator into one callable object.
"""

from groq import Groq
from dotenv import load_dotenv
from ingest import ingest_papers, PAPERS_DIR
from retriever import Retriever
from generator import generate

load_dotenv()


class RAGForge:
    """
    Main pipeline object.
    Initialize once (loads models, builds indexes).
    Call query() as many times as needed.
    """

    def __init__(self):
        print("=" * 50)
        print("Initializing RAG-Forge...")
        print("=" * 50 + "\n")

        # Step 1 — ingest all papers, get parent store
        print("Step 1: Ingesting papers...")
        self.parent_store = ingest_papers(PAPERS_DIR)

        # Step 2 — build retriever (loads models + BM25 index)
        print("\nStep 2: Building retriever...")
        self.retriever = Retriever(self.parent_store)

        # Step 3 — Groq client for generation + query expansion
        print("Step 3: Connecting to Groq...")
        self.groq_client = Groq()

        print("\nRAG-Forge ready.\n")


    def query(self, question: str, use_multiquery: bool = True) -> dict:
        """
        Full RAG pipeline:
        question → retrieval → generation → answer
        """
        print(f"Query: {question}")
        print("-" * 50)

        # Retrieve relevant parent chunks
        context_chunks = self.retriever.retrieve(
            question, self.groq_client, use_multiquery=use_multiquery
        )

        # Generate grounded answer
        result = generate(question, context_chunks, self.groq_client)

        return result


if __name__ == "__main__":
    pipeline = RAGForge()

    test_questions = [
        "What datasets did RAG models get evaluated on?",
        "How does DPR differ from sparse retrieval methods like BM25?",
        "What is HyDE and how does it improve retrieval?",
    ]

    for question in test_questions:
        result = pipeline.query(question)
        print(f"\nAnswer: {result['answer']}\n")
        print("=" * 50 + "\n")