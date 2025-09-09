from fastapi import APIRouter, UploadFile, File
from app.services import ingest

router = APIRouter()

@router.post("/")
async def upload(file: UploadFile = File(...)):
    await ingest.ingest_file(file)
    return {"status": "ingested", "filename": file.filename}
