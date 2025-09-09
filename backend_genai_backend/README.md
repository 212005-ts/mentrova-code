# GenAI Backend

## Run locally
```bash
uvicorn app.main:app --reload
```

## Run with Docker
```bash
docker build -t genai-backend .
docker run -p 8000:8000 genai-backend
```
