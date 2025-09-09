# app/tools/tool_manager.py
from app.tools.calculator import safe_eval
from app.tools.web_search import web_search

def use_tool(query: str) -> str | None:
    """Decide if a tool should be used based on the query."""
    q = query.lower()

    if any(word in q for word in ["calculate", "compute", "solve", "evaluate"]):
        expr = q.replace("calculate", "").replace("compute", "").strip()
        return "🧮 " + safe_eval(expr)

    if any(word in q for word in ["search", "look up", "find", "news", "today", "who is", "what is"]):
        return "🌐 " + web_search(query)

    return None
