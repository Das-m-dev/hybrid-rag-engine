import ast, csv, json, os, pathlib, re, sqlite3, time
from contextlib import closing
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import rag

app = FastAPI(title="Internal KB Q&A")
DB = os.getenv("USAGE_DB", "usage.db")
RESULTS = pathlib.Path("eval/results")
TESTSET = pathlib.Path("eval/testset.json")
MODES = ("hybrid", "vector", "keyword")
METRICS = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]

def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

with closing(db()) as _c, _c:
    _c.execute("""CREATE TABLE IF NOT EXISTS queries (id INTEGER PRIMARY KEY, ts REAL, question TEXT, mode TEXT,
                  latency_ms INTEGER, top_source TEXT, sources TEXT, answer TEXT, rating INTEGER)""")

class Ask(BaseModel):
    question: str
    k: int = 5
    mode: str | None = None   # hybrid | vector | keyword

class Feedback(BaseModel):
    id: int
    rating: int               # 1 = helpful, -1 = not helpful

@app.post("/api/ask")
def ask(body: Ask):
    if not body.question.strip():
        raise HTTPException(400, "Question is empty")
    if body.mode not in (None, *MODES):
        raise HTTPException(400, "mode must be hybrid, vector or keyword")
    t0 = time.time()
    try:
        res = rag.answer(body.question, body.k, body.mode)
    except Exception as e:  # Ollama down, model not pulled, etc.
        raise HTTPException(502, f"Model backend error: {e}. Is Ollama running and are the models pulled?")
    ms = int((time.time() - t0) * 1000)
    srcs = [s["source"] for s in res["sources"]]
    with closing(db()) as c, c:
        cur = c.execute("INSERT INTO queries (ts, question, mode, latency_ms, top_source, sources, answer) VALUES (?,?,?,?,?,?,?)",
                        (time.time(), body.question, body.mode or rag.MODE, ms, srcs[0] if srcs else None,
                         json.dumps(srcs, ensure_ascii=False), res["answer"]))
    return {**res, "id": cur.lastrowid, "latency_ms": ms}

@app.post("/api/feedback")
def feedback(body: Feedback):
    if body.rating not in (1, -1):
        raise HTTPException(400, "rating must be 1 or -1")
    with closing(db()) as c, c:
        if c.execute("UPDATE queries SET rating=? WHERE id=?", (body.rating, body.id)).rowcount == 0:
            raise HTTPException(404, "unknown question id")
    return {"ok": True}

@app.get("/api/health")
def health():
    return {"chunks": rag.collection().count(), "backend": rag.BACKEND, "search_mode": rag.MODE,
            "embed_model": rag.EMBED_MODEL, "llm_model": rag.LLM_MODEL}

@app.get("/api/stats")
def stats():
    metas = rag.collection().get(include=["metadatas"])["metadatas"]
    with closing(db()) as c:
        t = c.execute("SELECT COUNT(*) n, AVG(latency_ms) avg_ms, SUM(rating=1) up, SUM(rating=-1) down FROM queries").fetchone()
        recent = [dict(r) for r in c.execute(
            "SELECT id, ts, question, mode, latency_ms, top_source, rating FROM queries ORDER BY id DESC LIMIT 25")]
    return {"chunks": len(metas), "files": len({m["source"] for m in metas}), "questions": t["n"],
            "avg_ms": t["avg_ms"], "up": t["up"] or 0, "down": t["down"] or 0, "recent": recent}

@app.get("/api/eval/runs")
def eval_runs():
    runs = []
    for p in sorted(RESULTS.glob("*_summary.json"))[-30:]:
        d = json.loads(p.read_text(encoding="utf-8"))
        d["id"] = p.name.removesuffix("_summary.json")
        runs.append(d)
    return runs

@app.get("/api/eval/runs/{rid}/ragas/{mode}")
def eval_detail(rid: str, mode: str):
    if not re.fullmatch(r"\d{8}-\d{6}", rid) or mode not in MODES:
        raise HTTPException(400, "bad run id or mode")
    p = RESULTS / f"{rid}_ragas_{mode}.csv"
    if not p.exists():
        raise HTTPException(404, "No per-question file for this run")
    def num(r, k):
        try:
            v = float(r[k])
            return None if v != v else v
        except (KeyError, ValueError, TypeError):
            return None
    with p.open(encoding="utf-8-sig", newline="") as f:
        return [{"question": r.get("user_input"), "answer": r.get("response"), "reference": r.get("reference"),
                 "scores": {k: num(r, k) for k in METRICS}} for r in csv.DictReader(f)]

class EvalReq(BaseModel):
    k: int = 5
    modes: list[str] | None = None

@app.post("/api/eval/retrieval")
def run_retrieval_eval(body: EvalReq):
    """Instant, LLM-free check (hit@k, MRR). RAGAS runs from the command line: python evaluate.py"""
    import evaluate
    if not TESTSET.exists():
        raise HTTPException(404, "eval/testset.json not found")
    if rag.collection().count() == 0:
        raise HTTPException(400, "Index is empty. Run: python ingest.py <folder>")
    modes = body.modes or list(MODES)
    items = json.loads(TESTSET.read_text(encoding="utf-8-sig"))
    retrieval, misses = evaluate.retrieval_report(items, modes, body.k)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report = {"questions": len(items), "k": body.k, "retrieval": retrieval, "misses": misses, "ragas": {}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{stamp}_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**report, "id": stamp}

@app.get("/dashboard")
def dashboard():
    return FileResponse("static/dashboard.html")

app.mount("/", StaticFiles(directory="static", html=True), name="static")