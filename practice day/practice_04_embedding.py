from sentence_transformers import SentenceTransformer
import numpy as np
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Retrieval-Augmented Generation combines retrieval and generation.",
    "RAG models retrieve documents before generating an answer.",
    "I went to the gym this morning and did chest day.",
]

embeddings = model.encode(sentences)

def cosine_similarity(em1,em2):
    dotproduct = np.dot(em1,em2)
    norm = np.linalg.norm(em1)*np.linalg.norm(em2)
    return dotproduct/norm

sim_0_1 = cosine_similarity(embeddings[0], embeddings[1])
sim_0_2 = cosine_similarity(embeddings[0], embeddings[2])

print(f"\nSimilarity (RAG sentence vs RAG sentence): {sim_0_1:.4f}")
print(f"Similarity (RAG sentence vs gym sentence):  {sim_0_2:.4f}")