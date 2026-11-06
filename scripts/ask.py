# scripts/ask.py
import argparse
from copy import deepcopy
import json
import sys

from dotenv import load_dotenv
from openai import APIError

from ragdn.pipeline import RAG


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Tra cứu pháp luật doanh nghiệp.")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--mode", choices=("adaptive", "legacy"), default=None)
    parser.add_argument("--debug-retrieval", action="store_true", help="Hiện số đếm và ID truy xuất, không hiện prompt.")
    args = parser.parse_args()
    if args.question is None:
        print('Cách dùng: python scripts/ask.py "<câu hỏi>" [--mode adaptive|legacy] [--debug-retrieval]')
        raise SystemExit(1)

    question = args.question
    oversized = len(question) > 2000 and args.mode != "legacy"
    mode = "legacy" if oversized else args.mode
    if oversized:
        print("Câu hỏi vượt 2000 ký tự: dùng tra cứu legacy và giữ nguyên toàn bộ câu hỏi.", file=sys.stderr)
    print("\n[1] Đang nạp hệ thống RAG...")
    rag = RAG.load()

    print("[2] Đang xử lý câu hỏi...")
    try:
        res = rag.answer(question, mode=mode) if mode is not None else rag.answer(question)
    except (APIError, TimeoutError):
        print("Dịch vụ LLM đang lỗi hoặc chưa sẵn sàng, vui lòng kiểm tra kết nối và thử lại.", file=sys.stderr)
        raise SystemExit(1) from None
    if oversized:
        res = deepcopy(res)
        proc = res.setdefault("processing", {})
        proc.update(mode="baseline_fallback", route_reason="question_outside_adaptive_limit", status="not_assessed")
        proc["warnings"] = list(dict.fromkeys([*proc.get("warnings", []), "question_outside_adaptive_limit"]))

    print("\n" + "=" * 50)
    print("CÂU TRẢ LỜI:")
    print("=" * 50)
    print(res["disclaimer"])
    print(res["answer"])
    print("\n" + "=" * 50)
    print("NGUỒN TRÍCH DẪN:")
    for s in res["sources"]:
        print(f"- {s['chunk_id']} | {s['ten_vb']}, Điều {s['dieu']}")

    processing = res.get("processing") or {}
    if processing:
        print("\nPHẠM VI TRẢ LỜI:")
        print(f"- Chế độ: {processing.get('mode', 'legacy')}; trạng thái: {processing.get('status', 'not_assessed')}")
        for issue in processing.get("issues", []):
            print(f"- {issue.get('title', 'Vấn đề')}: {issue.get('status', 'not_assessed')}")
        for info in processing.get("missing_information", []):
            print(f"- Cần làm rõ: {info.get('description', '')}")
        for warning in processing.get("warnings", []):
            print(f"- Lưu ý: {warning}")
    if res.get("partial"):
        print("- Câu trả lời mới giải quyết một phần yêu cầu.")
    if args.debug_retrieval:
        diagnostics = res.get("diagnostics") or {}
        safe = {key: diagnostics[key] for key in ("variant", "context_tokens", "context_method", "prompt_tokens",
                                                 "groups", "seen_spans", "pool_excluded_parent_ids") if key in diagnostics}
        safe["budget"] = processing.get("budget", {})
        safe["reader_context"] = processing.get("reader_context", {})
        print("\nCHẨN ĐOÁN TRUY XUẤT:")
        print(json.dumps(safe, ensure_ascii=False))

if __name__ == "__main__":
    main()
