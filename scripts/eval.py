# scripts/eval.py
import json
from collections import defaultdict

from ragdn.pipeline import RAG


def eval_retrieval(golden: list[dict], rag: RAG, k: int = 5):
    agg = defaultdict(lambda: {"n": 0, "hit": 0, "recall": 0.0, "mrr": 0.0})
    for g in golden:
        if g.get("should_refuse", False):
            continue
        hits = rag.retrieve(g["question"])["hits"]
        keys = list(dict.fromkeys((rag.retriever.chunks[i]["doc_id"], rag.retriever.chunks[i]["dieu"]) for i in hits))[:k]
        gold = {tuple(x.split(":", 1)) for x in g["gold"]}
        found = [kk in gold for kk in keys]
        for bucket in ("ALL", g.get("type", "normal")):
            a = agg[bucket]
            a["n"] += 1
            a["hit"] += any(found)
            a["recall"] += len(gold & set(keys)) / len(gold) if gold else 0.0
            a["mrr"] += next((1 / r for r, f in enumerate(found, 1) if f), 0.0)

    return {
        b: {
            f"hit@{k}": a["hit"] / a["n"] if a["n"] else 0,
            f"recall@{k}": a["recall"] / a["n"] if a["n"] else 0,
            f"mrr@{k}": a["mrr"] / a["n"] if a["n"] else 0,
            "n": a["n"],
        }
        for b, a in agg.items()
    }

if __name__ == "__main__":
    import pathlib
    golden_path = pathlib.Path("eval/golden_dev.jsonl")
    if not golden_path.exists():
        print("Chưa có file eval/golden_dev.jsonl để đánh giá.")
    else:
        rag = RAG.load()
        with open(golden_path, encoding="utf-8") as f:
            golden = [json.loads(line) for line in f]
        res = eval_retrieval(golden, rag)
        print(json.dumps(res, indent=2, ensure_ascii=False))
