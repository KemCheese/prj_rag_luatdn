# scripts/build_corpus.py
import json
from collections import Counter

from ragdn.chunk import check_corpus, chunk
from ragdn.config import CHUNKS_PATH, MANIFEST_PATH, PROCESSED_DIR
from ragdn.manifest import load_manifest
from ragdn.parse import parse, read_pdf


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(MANIFEST_PATH)
    all_chunks = []

    print(f"=== Bắt đầu Parse Corpus (Phiên bản: {manifest.corpus_version}) ===")
    for doc in manifest.documents:
        print(f"\n-> Đang xử lý: {doc.doc_id} ({doc.ten_ngan})")
        lines = read_pdf(doc.file)
        track_quotes = doc.parse.get("track_quotes", False)
        articles = parse(lines, doc.doc_id, track_quotes=track_quotes)

        # Lưu articles trung gian
        art_path = PROCESSED_DIR / f"{doc.doc_id}.articles.json"
        art_path.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"   Trích xuất được: {len(articles)} Điều")

        # Thống kê chú thích
        notes_src = [n["source"] for a in articles for n in a["notes"]]
        if notes_src:
            print(f"   Top nguồn sửa đổi/hợp nhất: {Counter(notes_src).most_common(2)}")

        # Phân đoạn thành chunks
        doc_chunks = chunk(articles, doc.model_dump())
        print(f"   Tạo được: {len(doc_chunks)} chunks")
        all_chunks.extend(doc_chunks)

    # Kiểm tra tính toàn vẹn
    print("\n=== Kiểm tra Corpus (check_corpus) ===")
    errors = check_corpus(all_chunks, [d.model_dump() for d in manifest.documents])
    if errors:
        print("PHÁT HIỆN LỖI:")
        for err in errors:
            print(f"  [X] {err}")
        raise ValueError("Corpus chưa đạt chuẩn, vui lòng kiểm tra lại parser/manifest.")
    else:
        print(" -> check_corpus THÀNH CÔNG! Không phát hiện lỗi thiếu Điều hay trùng ID.")

    # Lưu chunks.jsonl
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in all_chunks)
    print(f"\nĐã lưu toàn bộ {len(all_chunks)} chunks vào: {CHUNKS_PATH}")

if __name__ == "__main__":
    main()
