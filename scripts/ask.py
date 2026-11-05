# scripts/ask.py
import sys

from dotenv import load_dotenv
from openai import APIError

from ragdn.pipeline import RAG


def main():
    load_dotenv()
    if len(sys.argv) < 2:
        print("Cách dùng: python scripts/ask.py \"<câu hỏi>\"")
        sys.exit(1)

    question = sys.argv[1]
    print("\n[1] Đang nạp hệ thống RAG...")
    rag = RAG.load()

    print(f"[2] Đang xử lý câu hỏi: '{question}'...")
    try:
        res = rag.answer(question)
    except APIError:
        print("Dịch vụ LLM đang lỗi hoặc chưa sẵn sàng, vui lòng kiểm tra kết nối và thử lại.", file=sys.stderr)
        raise SystemExit(1) from None

    print("\n" + "=" * 50)
    print("CÂU TRẢ LỜI:")
    print("=" * 50)
    print(res["disclaimer"])
    print(res["answer"])
    print("\n" + "=" * 50)
    print("NGUỒN TRÍCH DẪN:")
    for s in res["sources"]:
        print(f"- {s['chunk_id']} | {s['ten_vb']}, Điều {s['dieu']}")

if __name__ == "__main__":
    main()
