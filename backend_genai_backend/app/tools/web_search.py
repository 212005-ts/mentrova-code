# app/tools/web_search.py
import requests

def web_search(query: str) -> str:
    """Simple DuckDuckGo search."""
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
        if "AbstractText" in data and data["AbstractText"]:
            return data["AbstractText"]
        elif "RelatedTopics" in data and data["RelatedTopics"]:
            return data["RelatedTopics"][0].get("Text", "No results.")
        else:
            return "No results found."
    except Exception as e:
        return f" Web search failed: {str(e)}"
