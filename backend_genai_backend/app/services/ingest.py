import faiss, os
from PyPDF2 import PdfReader
from app.services import embeddings

INDEX_PATH = "vector.index"
_dimension = 384
_index = faiss.IndexFlatL2(_dimension)

async def ingest_file(file):
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    if file.filename.endswith(".pdf"):
        reader = PdfReader(path)
        texts = [page.extract_text() for page in reader.pages if page.extract_text()]
    else:
        with open(path, "r", encoding="utf-8") as f:
            texts = f.readlines()

    for t in texts:
        if t.strip():
            vec = embeddings.get_embedding(t)
            _index.add([vec])
    faiss.write_index(_index, INDEX_PATH)
