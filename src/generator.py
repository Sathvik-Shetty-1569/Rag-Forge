"""
generator.py — RAG-Forge generation
Formats retrieved context + calls Groq with strict grounding prompt.
"""

from groq import Groq


GROQ_MODEL = "llama-3.1-8b-instant"


def generate(query: str, context_chunks: list[str], client: Groq) -> dict:
    """
    Generate a grounded answer from retrieved context.
    Returns dict with answer + source chunks for citation.
    """
    context = "\n\n".join([
        f"[Source {i+1}]:\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    ])

    system_prompt = f"""You are a research assistant specializing in NLP and AI papers.
Answer the user's question using ONLY the sources provided below.
Be specific — cite which source supports each claim using [Source N] notation.
If the sources don't contain enough information to answer, say exactly:
"I don't have enough context to answer this question."
Do not use any outside knowledge beyond what is in the sources.

Sources:
{context}"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query},
        ],
        temperature=0.1,  # low temperature = more faithful, less creative
    )

    return {
        "answer":  response.choices[0].message.content,
        "sources": context_chunks,
    }