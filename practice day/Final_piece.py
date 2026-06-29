from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from pypdf import PdfReader
import re
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

def text_filtering(file):
    fulltext = []
    reader = PdfReader(str(file))
    
    for page in reader.pages:
        text = page.extract_text()
        if text :
            fulltext.append(text)
        
    raw_text = "\n".join(fulltext)
    cleaned_text = re.sub(r'[ \t\n]+'," ",raw_text)
    cleaned = cleaned_text.replace("ﬁ", "fi").replace("ﬂ", "fl")
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
    
def build_child_parent_chunk(file):
    parent_chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 100,
        separators=[" ."," ",""]
    )
    child_chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 20,
        separators=[" ."," ",""]
    )
    all_parent_chunks = {}
    all_child_chunks = []
    for parent_chunk in parent_chunk_splitter.split_text(file):
        if is_garbage_chunk(parent_chunk):
            continue
        parents_id = str(uuid.uuid4())
        all_parent_chunks[parents_id] = parent_chunk
        
        for child_chunk in child_chunk_splitter.split_text(parent_chunk):
            if not is_garbage_chunk(child_chunk):
                all_child_chunks.append({"text" : child_chunk,"pid": parents_id})
    return all_parent_chunks, all_child_chunks
            
            
def storing(child_text , groq , embedding_model,collection,pid):
    embeddings = embedding_model.encode(child_text)
    collection.upsert(
        ids =[f'doc_{i}' for i in range(len(child_text))],
        embeddings = embeddings.tolist(),
        documents = child_text,
        metadatas = [{"pid": p} for p in pid]
    )
    return child_text,[{"pid": p} for p in pid]
    
def tokenize(chunk):
    return chunk.lower().split()

def _generate_queries(self, query: str, groq_client) -> list[str]:

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

def dense_retriever(query,embedding_model,collection):
    
    query_embedding = embedding_model.encode([query])
    result = collection.query(
        query_embeddings = query_embedding.tolist(),
        n_results = 20,
        include = ['metadatas','documents','distances']
    )
    output = []
    for x in (zip(result['distances'][0],result['document'][0],result['metadata'][0])):
        scores = 1 - x[0]
        output.append(scores,x[1],x[2])
    return output
    
    
def bm25_retriever(query,bm25_model,documents,metadatas):
    scores = bm25_model.get_scores(tokenize(query))
    sort_scores = sorted(enumerate(scores),key=lambda x: x[1] ,reverse=True)
    output = []
    for i,rank in sort_scores[:20]:
        output.append(rank,documents[i],metadatas[i]['pid'])
    return output

def rank_reciprocal_retriever(dense,bm25):
    rrs_chunk ={}
    chunk_map = {}
    
    for i,_,doc,metadata in enumerate(dense,1):
        key = doc[:50]
        rank = rrs_chunk.get(key,0) + (1/(i+60))
        chunk_map[key] = (doc,metadata)
    
    for i,_,doc,metadata in enumerate(bm25,1):
        key = doc[:50]
        rrs_chunk[key] = rrs_chunk.get(key,0) + (1/(i+60))
        chunk_map[key] = (doc,metadata)
    
    sorted_key = sorted(rrs_chunk, key = rrs_chunk.get,reverse=True)
    output = []
    for k in sorted_key :
        output.append(rrs_chunk[k],chunk_map[k][0],chunk_map[k][1])
    return output
        
        
    
    
    
def retriever(queries,embedding_model,bm25_model,collection,documents,metadatas,cross_encoder_model,all_parent_chunk):
    
    all_candidates = {}
    for query in queries:
        dense_search = dense_retriever(query,embedding_model,collection)
        bm_search = bm25_retriever(query,bm25_model,documents,metadatas)
        rank_reciprocal = rank_reciprocal_retriever(dense_search,bm_search)
        
        for score,doc,metadata in rank_reciprocal[:20]:
            key = doc[:50]
            if key not in all_candidates or all_candidates[key] < score:
                all_candidates[key] = (score,doc,metadata)
                
    
    sorted_all_candidate = sorted(all_candidates.values(), key = lambda x : x[0], reverse=True)[:20]   
    candidate_texts = [t for _, t, _ in sorted_all_candidate]
    candidate_pids  = [p for _, _, p in sorted_all_candidate]
    
    pairs  = [(query, text) for text in candidate_texts]
    scores = cross_encoder_model.predict(pairs)
    ranked = sorted(zip(scores, candidate_texts, candidate_pids),reverse=True)[:5]
    
    seen_pids = set()
    parent_chunks = []

    for _, child_text, pid in ranked:
        if pid not in seen_pids:
            seen_pids.add(pid)
            parent_text = all_parent_chunks.get(pid)
            parent_chunks.append(parent_text)
    print(f"Parent chunks returned: {len(parent_chunks)}\n")
    return parent_chunks
    

if __name__ == "__main__":
    file_dir = Path('data\papers')
    crossencoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pdfs = list(file_dir.glob("*.pdf"))
    all_parent_chunks = {}
    all_child_chunks = []
    for pdf in pdfs:
        clean_text = text_filtering(pdf)
        parent_chunks , child_chunks = build_child_parent_chunk(clean_text)
        all_parent_chunks.update(parent_chunks)
        all_child_chunks.extend(child_chunks)
    
    child_text = [c["text"] for c in all_child_chunks]
    pid = [c["pid"] for c in all_child_chunks]
    groq_model = Groq()
    
    embedding_model = SentenceTransformer('all-miniLM-L6-v2')
    bm25_model = BM25Okapi(tokenize(c) for c in child_text)
    client = chromadb.PersistentClient(path = "./practice_chroma")
    collection = client.get_or_create_collection(name="MY_DOC",metadata={'hnsw:space':'cosine'})
    documents , metadatas = storing(child_text,groq_model,embedding_model,collection,pid)
    
    query = "What is rag ?"
    query_embedding = embedding_model.encode(query)
    result = collection.query(
        query_embeddings = query_embedding.tolist(),
        n_results = 3,
        include = ['distances','documents','metadatas']
    )
    
    
    while True:
        print("=" * 50 + "\n")
        query = input("How can I help you ?\n")
        
        if query == "exit":
            break
        
        queries = _generate_queries(query,groq_model)
        parent_chunk = retriever(query,embedding_model,bm25_model,collection,documents,metadatas,crossencoder_model,all_parent_chunks)
        result = generate(query, parent_chunks, groq_model)
        print(result['answer'])
        print("=" * 50 + "\n")
    
        
        
    
    
    
    