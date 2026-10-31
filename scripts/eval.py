# scripts/eval.py
import json
import pathlib
import sys
from collections import defaultdict

# Đảm bảo terminal Windows không bị lỗi font khi in ký tự tiếng Việt
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
    golden_path = pathlib.Path("eval/golden_dev_test_retrieval.jsonl")
    output_path = pathlib.Path("eval/golden_check.jsonl")

    if not golden_path.exists():
        print("Chưa có file eval/golden_dev_test_retrieval.jsonl để đánh giá.")
    else:
        rag = RAG.load()
        with open(golden_path, encoding="utf-8") as f:
            golden = [json.loads(line) for line in f]

        print(f"Đang chạy đánh giá {len(golden)} câu hỏi...")
        res = eval_retrieval(golden, rag)

        tmp_output_path = output_path.with_suffix(".tmp")
        with open(tmp_output_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

        tmp_output_path.replace(output_path)

        print(f"\n[DONE] Đã lưu kết quả đánh giá vào: {output_path}")
        if "ALL" in res:
            all_m = res["ALL"]
            print(
                f"Tóm tắt (ALL): Hit@5 = {all_m['hit@5']:.2%} | "
                f"Recall@5 = {all_m['recall@5']:.2%} | "
                f"MRR@5 = {all_m['mrr@5']:.4f} (N={all_m['n']})"
            )
