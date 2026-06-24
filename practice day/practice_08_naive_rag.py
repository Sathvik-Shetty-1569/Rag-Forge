from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from pathlib import Path
import chromadb
from groq import Groq
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv() 

def text_cleaning(file:Path)->str:
    fulltext = []
    reader = PdfReader(file)
    
    for doc in reader.pages:
        text = doc.extract_text()
        if text :
            fulltext.append(text)
    
    raw = "\n".join(fulltext)
    cleaned_text = re.sub('[ \t\n]+'," ",raw)
    cleaned = cleaned_text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return cleaned

def split_texting(chunks:str , chunk_size:int = 500, chunk_overlapped:int = 50)-> list[str]:
    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_overlap = chunk_overlapped,
        chunk_size = chunk_size,
        separators=[". ", " ", ""]
    )
    chunks = chunk_splitter.split_text(chunks)
    return chunks

def query_retrieval(query:str ,collection, model,k):
    query_embedding = model.encode([query])
    
    result = collection.query(
        query_embeddings = query_embedding,
        n_results= k
    )
    
    return result['documents'][0]

def generate(query: str, chunks: list[str], client: Groq) -> str:
    # Format retrieved chunks into a readable context block
    context = "\n\n".join([f"[chunk {i+1}]: {chunk}"
                            for i, chunk in enumerate(chunks)])

    system_prompt = """You are a research assistant specializing in NLP papers.
Answer the user's question using ONLY the context provided below.
If the context doesn't contain enough information, say exactly:
'I don't have enough context to answer this.'
Do not use any outside knowledge.

Context:
{context}""".format(context=context)

    response = client.chat.completions.create(
        model= "llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":

    model = SentenceTransformer('all-MiniLM-L6-v2')
    path = Path('data/papers/2005.11401v4.pdf')
    
    raw_text = text_cleaning(path)
    
    groq_client = Groq()
    client = chromadb.PersistentClient(path='./chroma')
    collection = client.get_or_create_collection(name='my_docs',metadata={"hnsw:space": "cosine"})
    
    
    chunks = split_texting(raw_text)
    
    embeddings = model.encode(chunks)
    
    
    collection.upsert(
        ids=[f"doc_{i}" for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=chunks
    )
    
    
    print("\nNaive RAG pipeline ready. Type 'exit' to quit.\n")

    while True:
        
        query = input("Your question: ").strip()
        
        if query == "exit":
            break
        
        if not query:
            continue
        
        retrieved_chunks = query_retrieval(query, collection, model, k=3)
        answer = generate(query, retrieved_chunks, groq_client)

        print(f"\nAnswer: {answer}\n")
        print("-" * 60 + "\n")
        
        
        
        
        