"""
retriever.py — RAG-Forge retrieval pipeline
Hybrid search (dense + BM25 + RRF) → cross-encoder reranking → parent lookup
"""

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

from ingest import CHROMA_PATH, COLLECTION, EMBED_MODEL


DENSE_MODEL    = EMBED_MODEL
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_K    = 20   # wide net for hybrid retrieval
RERANK_TOP_K   = 5    # after reranking, keep best 5
RRF_K          = 60   # RRF dampening constant


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class Retriever:
    """
    Wraps all retrieval logic in one clean object.
    Initialize once, call retrieve() many times.
    """

    def __init__(self, parent_store: dict):
        self.parent_store  = parent_store
        self.embed_model   = SentenceTransformer(DENSE_MODEL)
        self.cross_encoder = CrossEncoder(RERANK_MODEL)

        # Load Chroma collection
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        # Build BM25 index from all stored child chunks
        print("Building BM25 index...")
        all_docs = self.collection.get(include=["documents", "metadatas"])
        self.all_texts     = all_docs["documents"]
        self.all_metadatas = all_docs["metadatas"]
        self.all_ids       = all_docs["ids"]
        tokenized = [tokenize(t) for t in self.all_texts]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 index ready — {len(self.all_texts)} chunks indexed\n")


    def _dense_retrieve(self, query: str, k: int) -> list[tuple]:
        """Dense retrieval via Chroma. Returns [(score, chunk_text, parent_id)]"""
        query_emb = self.embed_model.encode([query])
        results = self.collection.query(
            query_embeddings=query_emb.tolist(),
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Chroma cosine distance → similarity (1 - distance)
            score = 1 - dist
            output.append((score, doc, meta["parent_id"]))
        return output


    def _bm25_retrieve(self, query: str, k: int) -> list[tuple]:
        """BM25 retrieval. Returns [(score, chunk_text, parent_id)]"""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        output = []
        for idx, score in ranked[:k]:
            output.append((score, self.all_texts[idx],
                           self.all_metadatas[idx]["parent_id"]))
        return output


    def _reciprocal_rank_fusion(self, dense_results, bm25_results) -> list[tuple]:
        """
        Merge dense + BM25 rankings using RRF.
        Ignores raw scores — only rank position matters.
        Returns [(rrf_score, chunk_text, parent_id)]
        """
        rrf_scores = {}
        chunk_map  = {}

        for rank, (_, text, pid) in enumerate(dense_results, 1):
            key = text[:50]  # use chunk prefix as dedup key
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (RRF_K + rank)
            chunk_map[key]  = (text, pid)

        for rank, (_, text, pid) in enumerate(bm25_results, 1):
            key = text[:50]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (RRF_K + rank)
            chunk_map[key]  = (text, pid)

        sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return [(rrf_scores[k], chunk_map[k][0], chunk_map[k][1])
                for k in sorted_keys]


    def _generate_queries(self, query: str, groq_client) -> list[str]:
        """Multi-query expansion — generate 3 rephrasings."""
        from groq import Groq
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate exactly 3 rephrasings of the user's question. "
                        "Use different vocabulary but preserve meaning. "
                        "Return ONLY the 3 questions, one per line, no numbering."
                    )
                },
                {"role": "user", "content": query}
            ]
        )
        raw = response.choices[0].message.content
        rephrasings = [l.strip() for l in raw.split("\n") if l.strip()]
        return [query] + rephrasings[:3]


    def retrieve(self, query: str, groq_client,
                 use_multiquery: bool = True) -> list[str]:
        """
        Full retrieval pipeline:
        1. Multi-query expansion
        2. Hybrid search (dense + BM25 + RRF) per query
        3. Merge all results
        4. Cross-encoder reranking
        5. Parent lookup — swap child chunks for full parent context
        """

        # Step 1 — query expansion
        queries = self._generate_queries(query, groq_client) \
                  if use_multiquery else [query]

        print(f"Queries used: {len(queries)}")

        # Step 2+3 — hybrid retrieval for each query, merge
        all_candidates = {}  # key → (rrf_score, text, parent_id)

        for q in queries:
            dense  = self._dense_retrieve(q, k=CANDIDATE_K)
            bm25   = self._bm25_retrieve(q, k=CANDIDATE_K)
            fused  = self._reciprocal_rank_fusion(dense, bm25)

            for rrf_score, text, pid in fused[:CANDIDATE_K]:
                key = text[:50]
                if key not in all_candidates or \
                   rrf_score > all_candidates[key][0]:
                    all_candidates[key] = (rrf_score, text, pid)

        # Sort merged pool by RRF score
        sorted_candidates = sorted(
            all_candidates.values(), key=lambda x: x[0], reverse=True
        )[:CANDIDATE_K]

        candidate_texts = [t for _, t, _ in sorted_candidates]
        candidate_pids  = [p for _, _, p in sorted_candidates]

        print(f"Candidates after hybrid+RRF: {len(candidate_texts)}")

        # Step 4 — cross-encoder reranking
        pairs  = [(query, text) for text in candidate_texts]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, candidate_texts, candidate_pids),
                        reverse=True)[:RERANK_TOP_K]

        print(f"After reranking: top {len(ranked)} chunks")

        # Step 5 — parent lookup
        seen_pids = set()
        parent_chunks = []

        for _, child_text, pid in ranked:
            if pid not in seen_pids:
                seen_pids.add(pid)
                parent_text = self.parent_store.get(pid, child_text)
                parent_chunks.append(parent_text)

        print(f"Parent chunks returned: {len(parent_chunks)}\n")
        return parent_chunks