import argparse, json, os, pathlib, time, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import rag

MODES = ["hybrid", "vector", "keyword"]

def retrieval_report(items, modes, k):
    out = {}
    for mode in modes:
        hits, rr = 0, 0.0
        for it in items:
            srcs = [h["source"] for h in rag.retrieve(it["question"], k, mode)]
            if it["source"] in srcs:
                hits += 1
                rr += 1 / (srcs.index(it["source"]) + 1)
        out[mode] = {f"hit@{k}": round(hits / len(items), 3), "mrr": round(rr / len(items), 3)}
    return out

def make_judge():
    """Local Ollama model as RAGAS judge + bge-m3 for answer-relevancy embeddings."""
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    model = os.getenv("JUDGE_MODEL", rag.LLM_MODEL)
    llm = LangchainLLMWrapper(ChatOllama(model=model, base_url=rag.OLLAMA, temperature=0, num_ctx=8192, format="json"))
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=rag.EMBED_MODEL, base_url=rag.OLLAMA))
    return llm, emb

def run_ragas(items, mode, k, llm, emb):
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
    from ragas.run_config import RunConfig
    rows = []
    for it in items:
        res = rag.answer(it["question"], k, mode)
        rows.append({"user_input": it["question"], "response": res["answer"], "reference": it["reference"],
                     "retrieved_contexts": [s["text"] for s in res["sources"]]})
    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()]
    result = evaluate(EvaluationDataset.from_list(rows), metrics=metrics, llm=llm, embeddings=emb,
                      run_config=RunConfig(timeout=900, max_workers=1, max_retries=3), show_progress=True)
    df = result.to_pandas()
    summary = {}
    for m in metrics:
        if m.name in df:
            summary[m.name] = {"mean": None if df[m.name].isna().all() else round(float(df[m.name].mean()), 3),
                               "failed": int(df[m.name].isna().sum())}  # failed = judge output could not be parsed
    return df, summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="eval/testset.json")
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES, help="modes for the retrieval report")
    ap.add_argument("--ragas-modes", nargs="+", default=["hybrid"], choices=MODES, help="modes scored by RAGAS")
    ap.add_argument("--skip-ragas", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke test)")
    a = ap.parse_args()

    items = json.loads(pathlib.Path(a.testset).read_text(encoding="utf-8"))
    items = items[:a.limit] if a.limit else items
    if rag.collection().count() == 0:
        raise SystemExit("Index is empty. Run: python ingest.py <folder>")
    out_dir = pathlib.Path("eval/results"); out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report = {"questions": len(items), "k": a.k, "retrieval": retrieval_report(items, a.modes, a.k), "ragas": {}}

    print(f"\nRetrieval ({len(items)} questions, k={a.k})")
    for mode, r in report["retrieval"].items():
        print(f"  {mode:8} " + "  ".join(f"{n}={v}" for n, v in r.items()))

    if not a.skip_ragas:
        llm, emb = make_judge()
        print(f"\nRAGAS judge: {os.getenv('JUDGE_MODEL', rag.LLM_MODEL)} (about {11 * len(items)} LLM calls per mode, run sequentially)")
        for mode in a.ragas_modes:
            df, summary = run_ragas(items, mode, a.k, llm, emb)
            report["ragas"][mode] = summary
            df.to_csv(out_dir / f"{stamp}_ragas_{mode}.csv", index=False, encoding="utf-8-sig")
            print(f"\nRAGAS [{mode}]")
            for name, s in summary.items():
                print(f"  {name:38} mean={s['mean']}  failed={s['failed']}/{len(items)}")

    (out_dir / f"{stamp}_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out_dir}/{stamp}_*")

if __name__ == "__main__":
    main()