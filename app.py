from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import rag

app = FastAPI(title="Local RAG")

class Ask(BaseModel):
    question: str
    k: int = 5
    mode: str | None = None   # hybrid | vector | keyword

@app.post("/api/ask")
def ask(body: Ask):
    if not body.question.strip():
        raise HTTPException(400, "Question is empty")
    if body.mode not in (None, "hybrid", "vector", "keyword"):
        raise HTTPException(400, "mode must be hybrid, vector or keyword")
    try:
        return rag.answer(body.question, body.k, body.mode)
    except Exception as e:  # Ollama down, model not pulled, etc.
        raise HTTPException(502, f"Model backend error: {e}. Is Ollama running and are the models pulled?")

@app.get("/api/health")
def health():
    return {"chunks": rag.collection().count(), "backend": rag.BACKEND, "search_mode": rag.MODE,
            "embed_model": rag.EMBED_MODEL, "llm_model": rag.LLM_MODEL}

app.mount("/", StaticFiles(directory="static", html=True), name="static")