# Local RAG app

Ask questions over your own documents (policies, code docs, PDFs). Everything runs on your machine via Ollama + ChromaDB.

## Run
```bash
ollama pull bge-m3 && ollama pull qwen2.5:7b      
pip install -r requirements.txt
python ingest.py docs                               
uvicorn app:app --port 8000                   
```

## Use real documents
```bash
git clone --depth 1 https://github.com/fastapi/fastapi ../fastapi
python ingest.py ../fastapi/docs/en/docs            # then ask "How do I declare a query parameter?"
```
Supports .md .txt .rst .pdf and common code files. Re-running `ingest.py` rebuilds the index.

## Config (env vars)
`EMBED_MODEL`, `LLM_MODEL`, `OLLAMA_URL`, `DB_DIR`. Set `RAG_BACKEND=offline` to test the pipeline without Ollama (keyword-hash embeddings, no LLM).

## How it works
`ingest.py` chunks files (~900 chars, paragraph-aware, keeps headings) -> embeds with bge-m3 -> stores in ChromaDB (cosine).
`rag.py` embeds the question, retrieves top-k chunks, and prompts the LLM to answer only from them with [n] citations.
