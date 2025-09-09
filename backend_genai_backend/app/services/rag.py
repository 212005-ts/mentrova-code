import faiss, os
from app.services.llm import phi2

INDEX_PATH = "vector.index"
_dimension = 384

if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = faiss.IndexFlatL2(_dimension)

def answer_query(query: str) -> str:
    vec = embeddings.get_embedding(query)
    if index.ntotal == 0:
        return "No documents ingested yet. Please upload."
    D, I = index.search([vec], k=3)
    return f"Retrieved docs idx: {I.tolist()} (mock answer for: {query})"