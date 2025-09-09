from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import time

from app.services.run_phi2 import phi2      # Phi-2 singleton
from app.services.run_gemma import gemma    # Gemma singleton
from app.services.run_qwen import qwen      # Qwen singleton
# from app.services.run_llama import llama2  # 🚧 coming soon

from app.tools.tool_manager import use_tool  # ✅ NEW

app = FastAPI()

# --------------------------
# In-memory session storage
# --------------------------
chat_histories: Dict[str, List[Dict[str, str]]] = {}

# --------------------------
# System Prompts
# --------------------------
STUDENT_PROMPT = """You are Mentrova, a personal AI tutor for students.
- Explain concepts clearly, step by step, in simple words.
- Use analogies, examples, and small exercises when useful.
- Be patient, encouraging, and motivational.
- Never mention OpenAI or your training data.
"""

CORPORATE_PROMPT = """You are Mentrova, an AI consultant for professionals and corporate employees.
- Provide concise, professional, and actionable responses.
- Use formal tone and structured bullet points when needed.
- Focus on productivity, efficiency, and clarity.
- Never mention OpenAI or your training data.
"""

# --------------------------
# Request Model
# --------------------------
class ChatRequest(BaseModel):
    session_id: str
    query: str
    mode: str   # "student" or "corporate"
    model: Optional[str] = Field(default="gemma", description="Defaults to gemma if not provided")

# --------------------------
# Helpers
# --------------------------
def build_prompt(session_id: str, system_prompt: str, user_query: str) -> str:
    history = chat_histories.get(session_id, [])
    formatted_history = "\n".join([f"{turn['role'].capitalize()}: {turn['content']}" for turn in history])
    return f"{system_prompt}\n{formatted_history}\nUser: {user_query}\nAssistant:"

def pick_model(name: str):
    if name == "phi2":
        return phi2
    elif name == "gemma":
        return gemma
    elif name == "qwen":
        return qwen
    # elif name == "llama2":
    #     return llama2
    return None

# --------------------------
# Root
# --------------------------
@app.get("/")
async def root():
    return {"message": "Backend is running 🚀 (Phi-2, Gemma, Qwen + Tools enabled — Gemma is default)"}

# --------------------------
# Chat (normal response)
# --------------------------
@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    mode = request.mode.lower()
    model_choice = (request.model or "gemma").lower()

    # Pick system prompt
    if mode == "student":
        system_prompt = STUDENT_PROMPT
    elif mode == "corporate":
        system_prompt = CORPORATE_PROMPT
    else:
        return {"error": "Invalid mode. Use 'student' or 'corporate'."}
    
    # ✅ Tool check
    tool_response = use_tool(request.query)
    if tool_response:
        if session_id not in chat_histories:
            chat_histories[session_id] = []
        chat_histories[session_id].append({"role": "user", "content": request.query})
        chat_histories[session_id].append({"role": "assistant", "content": tool_response})
        return {
            "mode": mode,
            "model": "tool",
            "response": tool_response,
            "history": chat_histories[session_id]
        }

    # Pick model
    model = pick_model(model_choice)
    if not model:
        return {"error": "Invalid model. Use 'phi2', 'gemma', or 'qwen'."}

    # Init session
    if session_id not in chat_histories:
        chat_histories[session_id] = []

    # Build prompt
    prompt = build_prompt(session_id, system_prompt, request.query)

    # Generate
    response = model.generate(prompt)

    # Save history
    chat_histories[session_id].append({"role": "user", "content": request.query})
    chat_histories[session_id].append({"role": "assistant", "content": response})

    return {
        "mode": mode,
        "model": model_choice,
        "response": response,
        "history": chat_histories[session_id]
    }

# --------------------------
# Chat (streaming response)
# --------------------------
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    session_id = request.session_id
    mode = request.mode.lower()
    model_choice = (request.model or "gemma").lower()

    if mode == "student":
        system_prompt = STUDENT_PROMPT
    elif mode == "corporate":
        system_prompt = CORPORATE_PROMPT
    else:
        return {"error": "Invalid mode. Use 'student' or 'corporate'."}

    # ✅ Tool check
    tool_response = use_tool(request.query)
    if tool_response:
        if session_id not in chat_histories:
            chat_histories[session_id] = []
        chat_histories[session_id].append({"role": "user", "content": request.query})
        chat_histories[session_id].append({"role": "assistant", "content": tool_response})

        async def tool_stream():
            yield (tool_response + "\n").encode("utf-8")
            yield b"\n[END]\n"

        return StreamingResponse(tool_stream(), media_type="text/plain; charset=utf-8")

    # Pick model
    model = pick_model(model_choice)
    if not model:
        return {"error": "Invalid model. Use 'phi2', 'gemma', or 'qwen'."}

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    prompt = build_prompt(session_id, system_prompt, request.query)

    async def generate_stream():
        full_response = ""
        try:
            if hasattr(model, "stream"):
                for chunk in model.stream(prompt):
                    full_response += chunk
                    yield (chunk + "\n").encode("utf-8")
            else:
                full_response = model.generate(prompt)
                for word in full_response.split():
                    yield (word + " ").encode("utf-8")
                    # time.sleep(0.03)
        finally:
            chat_histories[session_id].append({"role": "user", "content": request.query})
            chat_histories[session_id].append({"role": "assistant", "content": full_response})
            yield b"\n[END]\n"

    return StreamingResponse(generate_stream(), media_type="text/plain; charset=utf-8")

# --------------------------
# Benchmark (compare models or use tool)
# --------------------------
@app.post("/benchmark")
async def benchmark(request: ChatRequest):
    session_id = request.session_id
    mode = request.mode.lower()

    if mode == "student":
        system_prompt = STUDENT_PROMPT
    elif mode == "corporate":
        system_prompt = CORPORATE_PROMPT
    else:
        return {"error": "Invalid mode. Use 'student' or 'corporate'."}

    # ✅ Tool check first
    tool_response = use_tool(request.query)
    if tool_response:
        if session_id not in chat_histories:
            chat_histories[session_id] = []
        chat_histories[session_id].append({"role": "user", "content": request.query})
        chat_histories[session_id].append({"role": "assistant", "content": tool_response})
        return {
            "query": request.query,
            "mode": mode,
            "results": {"tool": tool_response}
        }

    # Otherwise run models
    prompt = build_prompt(session_id, system_prompt, request.query)

    results = {}
    for mname in ["phi2", "gemma", "qwen"]:
        m = pick_model(mname)
        try:
            results[mname] = m.generate(prompt)
        except Exception as e:
            results[mname] = f"[ERROR: {str(e)}]"

    return {
        "query": request.query,
        "mode": mode,
        "results": results
    }
