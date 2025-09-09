from fastapi import APIRouter
from pydantic import BaseModel
from app.services import rag

router = APIRouter()

class ChatRequest(BaseModel):
    query: str

@router.post("/")
def chat(request: ChatRequest):
    answer = rag.answer_query(request.query)
    return {"answer": answer}
