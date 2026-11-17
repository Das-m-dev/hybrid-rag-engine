"""Core RAG logic: embeddings, chunking, retrieval, generation. Local-only via Ollama."""
import hashlib, math, os, re
from collections import Counter
import numpy as np
import requests
import chromadb

OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")       
LLM_MODEL = os.getenv("LLM_MODEL", "gemma2:9b")         
BACKEND = os.getenv("RAG_BACKEND", "ollama")           
DB_DIR = os.getenv("DB_DIR", "./chroma_db")
MODE = os.getenv("SEARCH_MODE", "hybrid")           # "hybrid", "vector", "keyword"
RRF_K = 60                                           

_client = chromadb.PersistentClient(path=DB_DIR)

def collection():
    return _client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})

def embed(texts: list[str]) -> list[list[float]]:
    if BACKEND == "offline":  # hashed bag-of-words: only for tests without Ollama
        out = []
        for t in texts:
            v = np.zeros(512)
            for w in re.findall(r"\w+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 512] += 1
            out.append((v / (np.linalg.norm(v) or 1)).tolist())
        return out
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts}, timeout=300)
    r.raise_for_status()
    return r.json()["embeddings"]

def chunk(text: str, size=900, overlap=150):
    """Paragraph-aware chunks that remember the nearest markdown heading."""
    chunks, cur, heading, cur_heading = [], "", "", ""
    def flush():
        nonlocal cur
        if cur.strip():
            chunks.append((cur.strip(), cur_heading))
        cur = cur[-overlap:] if len(cur) > overlap else ""
    for para in re.split(r"\n\s*\n", text):
        if para.lstrip().startswith("#"):
            heading = para.strip().lstrip("# ").split("\n")[0]
        while len(para) > size:                      # hard-split giant paragraphs
            cur += "\n\n" + para[:size]; cur_heading = heading; flush(); para = para[size:]
        if len(cur) + len(para) > size and cur.strip():
            flush()
        if not cur.strip():
            cur_heading = heading
        cur += "\n\n" + para
    flush()
    return chunks

def tokenize(text: str) -> list[str]:
    """Words for Latin text; character bigrams for Japanese/CJK (no spaces to split on)."""
    toks = []
    for w in re.findall(r"[a-z0-9_]+|[\u3040-\u30ff\u3400-\u9fff]+", text.lower()):
        if re.match(r"[\u3040-\u30ff\u3400-\u9fff]", w):
            toks += [w[i:i + 2] for i in range(max(len(w) - 1, 1))]
        else:
            toks.append(w)
    return toks

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(docs)) or 1
        df = Counter(t for c in self.tf for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + .5) / (c + .5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        terms, out = set(tokenize(query)), []
        for tf, length in zip(self.tf, self.len):
            s = 0.0
            for t in terms:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            out.append(s)
        return out

_kw = {"key": None}
def _keyword_index():
    """BM25 index built from the Chroma collection; rebuilt automatically after re-ingest."""
    col = collection()
    key = (col.id, col.count())
    if _kw["key"] != key:
        d = col.get(include=["documents", "metadatas"])
        _kw.update(key=key, ids=d["ids"], docs=d["documents"], metas=d["metadatas"], bm25=BM25(d["documents"]))
    return _kw

def retrieve(question: str, k=5, mode=None):
    """Vector search + BM25 keyword search, merged with Reciprocal Rank Fusion."""
    mode = mode or MODE
    col = collection()
    n = col.count()
    if n == 0:
        return []
    pool = min(max(k * 4, 20), n)
    ranks, info = {}, {}
    if mode in ("hybrid", "vector"):
        res = col.query(query_embeddings=embed([question]), n_results=pool)
        for r, (i, d, m) in enumerate(zip(res["ids"][0], res["documents"][0], res["metadatas"][0]), 1):
            ranks.setdefault(i, {})["vector"] = r; info[i] = (d, m)
    if mode in ("hybrid", "keyword"):
        kw = _keyword_index()
        sc = kw["bm25"].scores(question)
        top = [j for j in sorted(range(len(sc)), key=lambda j: -sc[j])[:pool] if sc[j] > 0]
        for r, j in enumerate(top, 1):
            i = kw["ids"][j]
            ranks.setdefault(i, {})["keyword"] = r; info[i] = (kw["docs"][j], kw["metas"][j])
    rrf = lambda i: sum(1 / (RRF_K + r) for r in ranks[i].values())
    return [{"text": info[i][0], "source": info[i][1]["source"], "heading": info[i][1].get("heading", ""),
             "score": round(rrf(i), 4), "via": " + ".join(f"{m} #{r}" for m, r in ranks[i].items())}
            for i in sorted(ranks, key=rrf, reverse=True)[:k]]

SYSTEM = ("You answer questions using ONLY the numbered context excerpts. Cite the excerpts you use like [1] or [2]. "
          "If the context does not contain the answer, say you could not find it in the documents. "
          "Reply in the same language as the question.")

def answer(question: str, k=5, mode=None):
    hits = retrieve(question, k, mode)
    if not hits:
        return {"answer": "No documents are indexed yet. Run `python ingest.py <folder>` first.", "sources": []}
    if BACKEND == "offline":
        return {"answer": "(offline mode) Top passage:\n" + hits[0]["text"], "sources": hits}
    ctx = "\n\n".join(f"[{i+1}] ({h['source']}) {h['text']}" for i, h in enumerate(hits))
    r = requests.post(f"{OLLAMA}/api/chat", timeout=600, json={
        "model": LLM_MODEL, "stream": False, "options": {"temperature": 0.1},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}]})
    r.raise_for_status()
    return {"answer": r.json()["message"]["content"], "sources": hits}