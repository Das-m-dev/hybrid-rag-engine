# Local RAG app

Ask questions over your own documents (policies, code docs, PDFs). Everything runs on your machine via Ollama + ChromaDB.

## Run
```bash
ollama pull bge-m3 && ollama pull qwen2.5:7b      # one-time: embeddings + chat model (both handle EN/JA)
pip install -r requirements.txt
python ingest.py docs                               # index the sample policies (or any folder)
uvicorn app:app --port 8000                         # open http://localhost:8000
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

## Search modes
Hybrid (default) = vector + BM25 keyword search merged with Reciprocal Rank Fusion. Switch with the dropdown, the API `mode` field, or `SEARCH_MODE=vector|keyword|hybrid`. Restart the server after re-ingesting.

## Evaluation (RAGAS)
```bash
pip install -r requirements-eval.txt
python evaluate.py --skip-ragas                  # instant: hit@k and MRR for hybrid / vector / keyword
python evaluate.py --limit 3                     # smoke test of the RAGAS judge (faithfulness, relevancy, precision, recall)
python evaluate.py --ragas-modes hybrid vector   # full run, compare modes
```
Edit `eval/testset.json` (question, reference answer, expected source file) for your own documents.
Set `JUDGE_MODEL` to use a different Ollama model as the judge. Results go to `eval/results/`.