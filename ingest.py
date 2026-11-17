"""Index a folder of documents:  python ingest.py <folder>   (rebuilds the index each run)"""
import hashlib, pathlib, sys
from rag import chunk, embed, _client, collection

TEXT = {".md", ".txt", ".rst", ".py", ".js", ".ts", ".tsx", ".java", ".go", ".sql", ".json", ".yaml", ".yml", ".html"}

def read(p: pathlib.Path) -> str:
    if p.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(pg.extract_text() or "" for pg in PdfReader(str(p)).pages)
    return p.read_text(encoding="utf-8", errors="ignore")

def main(folder: str):
    root = pathlib.Path(folder)
    files = [p for p in root.rglob("*") if p.is_file() and (p.suffix.lower() in TEXT or p.suffix.lower() == ".pdf")
             and not any(part.startswith(".") for part in p.parts)]
    _client.delete_collection("docs") if "docs" in [c.name for c in _client.list_collections()] else None
    col, ids, docs, metas = collection(), [], [], []
    for p in files:
        for i, (text, heading) in enumerate(chunk(read(p))):
            ids.append(hashlib.sha1(f"{p}:{i}".encode()).hexdigest())
            docs.append(text); metas.append({"source": str(p.relative_to(root)), "heading": heading})
    for s in range(0, len(docs), 32):                      # embed in batches
        col.add(ids=ids[s:s+32], documents=docs[s:s+32], metadatas=metas[s:s+32], embeddings=embed(docs[s:s+32]))
        print(f"  indexed {min(s+32, len(docs))}/{len(docs)}", end="\r")
    print(f"\nDone: {len(files)} files -> {len(docs)} chunks")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs")
