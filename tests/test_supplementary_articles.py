# tests/test_supplementary_articles.py
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

import ragdn.pipeline as pipeline
from ragdn.chunk import check_corpus, chunk
from ragdn.config import CHUNKS_PATH, MANIFEST_PATH, META_PATH
from ragdn.index import build_bm25_indices, chunks_sha256
from ragdn.manifest import load_manifest
from ragdn.parse import parse
from ragdn.retrieval import Retriever


def _make_dummy_retriever(chunks: list[dict]) -> Retriever:
    n = len(chunks)
    emb = np.zeros((n, 4), dtype=np.float32)
    embedder = SimpleNamespace(encode=lambda q, normalize_embeddings=True: np.zeros(4, dtype=np.float32))
    reranker = SimpleNamespace(predict=lambda pairs: np.zeros(len(pairs), dtype=np.float32))
    bm25, bm25_noacc = build_bm25_indices(chunks)
    return Retriever(
        chunks=chunks,
        emb=emb,
        embedder=embedder,
        reranker=reranker,
        bm25=bm25,
        bm25_noacc=bm25_noacc,
    )


def test_parse_supplementary_articles_17a_17b_17d_17dd():
    """Tái hiện lỗi Điều 17a, 17b, 17d, 17đ, 17e bị bỏ sót và gộp nhầm vào Điều 17."""
    lines = [
        ("Chương II", 1),
        ("Điều 17. Quyền thành lập doanh nghiệp", 1),
        ("1. Khoản 1 điều 17.", 1),
        ("2. Khoản 2 điều 17.", 1),
        ("Điều 17a. Điều bổ sung thứ nhất", 1),
        ("1. Khoản 1 điều 17a.", 1),
        ("2. Khoản 2 điều 17a.", 1),
        ("Điều 17b. Điều bổ sung thứ hai", 2),
        ("1. Khoản 1 điều 17b.", 2),
        ("Điều 17c. Điều bổ sung thứ ba", 2),
        ("1. Khoản 1 điều 17c.", 2),
        ("Điều 17d. Điều bổ sung thứ tư", 2),
        ("1. Khoản 1 điều 17d.", 2),
        ("Điều 17đ. Điều bổ sung thứ năm", 2),
        ("1. Khoản 1 điều 17đ.", 2),
        ("Điều 17e. Điều bổ sung thứ sáu", 2),
        ("1. Khoản 1 điều 17e.", 2),
        ("Điều 18. Hợp đồng trước đăng ký doanh nghiệp", 3),
        ("1. Khoản 1 điều 18.", 3),
    ]
    articles = parse(lines, doc_id="test_doc")
    dieus = [a["dieu"] for a in articles]
    assert dieus == ["17", "17a", "17b", "17c", "17d", "17đ", "17e", "18"]

    # Đảm bảo khoản 2 của Điều 17 không bị nuốt nội dung của Điều 17a
    art_17 = articles[0]
    assert art_17["khoans"] == [
        {"k": "1", "lines": ["1. Khoản 1 điều 17."], "notes": []},
        {"k": "2", "lines": ["2. Khoản 2 điều 17."], "notes": []},
    ]

    art_17a = articles[1]
    assert art_17a["tieu_de"] == "Điều bổ sung thứ nhất"
    assert [k["k"] for k in art_17a["khoans"]] == ["1", "2"]


def test_parse_ignores_backward_or_duplicate_dieu_mentions():
    """Đảm bảo các dòng nhắc lại Điều cũ hoặc trùng số Điều trong nội dung không bị tách nhầm."""
    lines = [
        ("Điều 17. Quyền thành lập", 1),
        ("1. Nội dung khoản 1.", 1),
        ("Điều 17a. Điều bổ sung a", 1),
        ("1. Thực hiện theo quy định tại", 1),
        ("Điều 17. Trường hợp ngoại lệ áp dụng riêng.", 1),
        ("Điều 17a. Trường hợp ngoại lệ khác.", 1),
        ("Điều 17b. Điều bổ sung b", 1),
        ("1. Khoản 1 điều 17b.", 1),
        ("Điều 17a. Nhắc lại điều 17a ở đầu dòng.", 1),
    ]
    articles = parse(lines, doc_id="test_doc")
    assert [a["dieu"] for a in articles] == ["17", "17a", "17b"]
    assert len(articles[1]["khoans"][0]["lines"]) == 3
    assert len(articles[2]["khoans"][0]["lines"]) == 2


def test_parse_supplementary_khoans_ordering():
    """Kiểm tra thứ tự tăng dần của Khoản phụ (1 -> 1a -> 1b -> 1d -> 1đ -> 1e -> 2) và chặn lặp lùi."""
    lines = [
        ("Điều 8. Nghĩa vụ của doanh nghiệp", 1),
        ("1. Khoản 1 gốc.", 1),
        ("1a. Khoản 1a bổ sung.", 1),
        ("1b. Khoản 1b bổ sung.", 1),
        ("1a. Dòng nhắc lại khoản 1a bị ngắt dòng trong khoản 1b.", 1),
        ("1d. Khoản 1d bổ sung.", 1),
        ("1đ. Khoản 1đ bổ sung.", 1),
        ("1e. Khoản 1e bổ sung.", 1),
        ("2. Khoản 2 tiếp theo.", 1),
    ]
    articles = parse(lines, doc_id="test_doc")
    assert len(articles) == 1
    ks = [k["k"] for k in articles[0]["khoans"]]
    assert ks == ["1", "1a", "1b", "1d", "1đ", "1e", "2"]
    assert len(articles[0]["khoans"][2]["lines"]) == 2


def test_check_corpus_detects_swallowed_article():
    """Nếu Điều 1a (kể cả tiêu đề bắt đầu bằng nguyên âm có dấu như Á, Ủy...) bị nuốt vào Điều 1, check_corpus() phải phát hiện."""
    swallowed_chunks = [
        {
            "chunk_id": "test_doc:d1:p0",
            "doc_id": "test_doc",
            "dieu": "1",
            "text": "Luật Test | Chương I | Điều 1. Phạm vi\n1. Khoản 1 điều 1.\n2. Khoản 2 điều 1. Điều 1a. Áp dụng quy định bổ sung bị nuốt\n1. Khoản 1 điều 1a.",
        },
        {
            "chunk_id": "test_doc:d2:p0",
            "doc_id": "test_doc",
            "dieu": "2",
            "text": "Luật Test | Chương I | Điều 2. Hiệu lực\n1. Khoản 1 điều 2.",
        },
    ]
    manifest_docs = [{"doc_id": "test_doc", "so_dieu": 2}]
    errors = check_corpus(swallowed_chunks, manifest_docs)
    assert any("1a" in err for err in errors), f"Kỳ vọng phát hiện Điều 1a bị nuốt, nhưng nhận được: {errors}"


def test_check_corpus_detects_missing_supplementary_predecessor():
    """Nếu có Điều 1b mà thiếu Điều 1a (hoặc có hậu tố không hợp lệ như 1f, 1ab), check_corpus() phải báo lỗi."""
    chunks_missing_1a = [
        {
            "chunk_id": "test_doc:d1:p0",
            "doc_id": "test_doc",
            "dieu": "1",
            "text": "Luật Test | Chương I | Điều 1. Phạm vi\n1. Nội dung điều 1.",
        },
        {
            "chunk_id": "test_doc:d1b:p0",
            "doc_id": "test_doc",
            "dieu": "1b",
            "text": "Luật Test | Chương I | Điều 1b. Bổ sung\n1. Nội dung điều 1b.",
        },
        {
            "chunk_id": "test_doc:d1ab:p0",
            "doc_id": "test_doc",
            "dieu": "1ab",
            "text": "Luật Test | Chương I | Điều 1ab. Hậu tố sai\n1. Nội dung.",
        },
        {
            "chunk_id": "test_doc:d2:p0",
            "doc_id": "test_doc",
            "dieu": "2",
            "text": "Luật Test | Chương I | Điều 2. Hiệu lực\n1. Nội dung điều 2.",
        },
    ]
    manifest_docs = [{"doc_id": "test_doc", "so_dieu": 2}]
    errors = check_corpus(chunks_missing_1a, manifest_docs)
    assert any("1a" in err for err in errors), f"Kỳ vọng báo thiếu Điều 1a trước 1b, nhưng nhận được: {errors}"
    assert any("1ab" in err for err in errors), f"Kỳ vọng báo lỗi hậu tố không hợp lệ 1ab, nhưng nhận được: {errors}"


def test_article_lookup_supplementary_including_dd():
    """Kiểm tra Retriever.article_lookup() tìm đúng Điều 17a, 17d và Điều 17đ (có dấu, Telex dd, và fallback không dấu)."""
    lines = [
        ("Chương II", 1),
        ("Điều 17. Quyền thành lập", 1),
        ("1. Nội dung điều 17.", 1),
        ("Điều 17a. Quyền bổ sung a", 1),
        ("1. Nội dung điều 17a.", 1),
        ("Điều 17d. Quyền bổ sung d", 1),
        ("1. Nội dung điều 17d.", 1),
        ("Điều 17đ. Quyền bổ sung đ", 1),
        ("1. Nội dung điều 17đ.", 1),
    ]
    articles = parse(lines, doc_id="luat_dn_2020")
    meta = {
        "doc_id": "luat_dn_2020",
        "ten_ngan": "Luật Doanh nghiệp",
        "vai_tro": "chinh",
        "ngay_hieu_luc": "2021-01-01",
    }
    chunks = chunk(articles, meta)
    retriever = _make_dummy_retriever(chunks)

    hits_17a, _ = retriever.article_lookup("Quy định tại Điều 17a Luật Doanh nghiệp là gì?")
    assert [chunks[i]["dieu"] for i in hits_17a] == ["17a"]

    hits_17d, _ = retriever.article_lookup("Quy định tại Điều 17d Luật Doanh nghiệp là gì?")
    assert [chunks[i]["dieu"] for i in hits_17d] == ["17d"]

    hits_17dd, _ = retriever.article_lookup("Quy định tại Điều 17đ Luật Doanh nghiệp là gì?")
    assert [chunks[i]["dieu"] for i in hits_17dd] == ["17đ"]

    hits_telex, _ = retriever.article_lookup("Quy dinh tai dieu 17dd Luat Doanh nghiep la gi?")
    assert [chunks[i]["dieu"] for i in hits_telex] == ["17đ"]

    # Fallback không dấu khi văn bản chỉ có 17đ (không có 17d)
    chunks_only_dd = [c for c in chunks if c["dieu"] != "17d"]
    retriever_only_dd = _make_dummy_retriever(chunks_only_dd)
    hits_fallback, _ = retriever_only_dd.article_lookup("Quy dinh tai dieu 17d Luat Doanh nghiep la gi?")
    assert [chunks_only_dd[i]["dieu"] for i in hits_fallback] == ["17đ"]


def test_existing_corpus_integrity_unchanged():
    """Kiểm tra corpus và hash trên đĩa cả trước và sau khi người dùng chạy T3."""
    manifest = load_manifest(MANIFEST_PATH)
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    legacy_corpus = all("muc" not in c and "loai_hinh" not in c for c in chunks)
    exclude = {"cau_truc"} if legacy_corpus else set()
    assert check_corpus(chunks, [d.model_dump(exclude=exclude) for d in manifest.documents]) == []

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    assert len(chunks) == meta["n"]
    if legacy_corpus and "content_sha256" not in meta:
        ids = [c["chunk_id"] for c in chunks]
        assert len(chunks) == 676
        assert hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest() == meta["ids_sha256"]
    else:
        assert meta.get("content_sha256") == chunks_sha256(chunks)


def test_chunks_sha256_depends_on_id_and_row_order():
    chunks = [{"chunk_id": "a", "text": "Nội dung"}, {"chunk_id": "b", "text": "Nội dung"}]
    digest = chunks_sha256(chunks)
    assert chunks_sha256(chunks[::-1]) != digest
    chunks[0]["chunk_id"] = "c"
    assert chunks_sha256(chunks) != digest


@pytest.mark.parametrize(
    ("state", "error"),
    [
        ("legacy", "Index dùng định dạng cũ"),
        ("changed_text", "Dữ liệu chunk đã thay đổi"),
        ("wrong_rows", "khác số vector"),
        ("matching", None),
    ],
)
def test_rag_load_checks_index_content(tmp_path, monkeypatch, state, error):
    chunks = [{"chunk_id": "luat_dn_2020:d90:p0", "text": "Chương IV\nĐiều 90"}]
    meta = {"n": len(chunks), "content_sha256": chunks_sha256(chunks)}
    if state == "legacy":
        meta = {"n": len(chunks), "ids_sha256": "old"}
    elif state == "changed_text":
        chunks[0]["text"] = "Chương III\nĐiều 90"

    chunks_path = tmp_path / "chunks.jsonl"
    meta_path = tmp_path / "meta.json"
    emb_path = tmp_path / "emb.npy"
    chunks_path.write_text(json.dumps(chunks[0], ensure_ascii=False) + "\n", encoding="utf-8")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    np.save(emb_path, np.zeros((2 if state == "wrong_rows" else 1, 4)))
    monkeypatch.setattr(pipeline, "CHUNKS_PATH", chunks_path)
    monkeypatch.setattr(pipeline, "META_PATH", meta_path)
    monkeypatch.setattr(pipeline, "EMB_PATH", emb_path)

    model_calls = []

    def load_model(name, **kwargs):
        model_calls.append(name)

    monkeypatch.setattr(pipeline, "SentenceTransformer", load_model)
    monkeypatch.setattr(pipeline, "CrossEncoder", load_model)
    if error:
        with pytest.raises(ValueError, match=error) as exc:
            pipeline.RAG.load()
        assert "scripts/build_index.py" in str(exc.value)
        assert model_calls == []
    else:
        monkeypatch.setattr(pipeline, "build_bm25_indices", lambda chunks: (None, None))
        monkeypatch.setattr(pipeline, "Retriever", SimpleNamespace)
        assert pipeline.RAG.load().retriever.chunks == chunks
        assert model_calls == [pipeline.MODEL_EMBEDDING, pipeline.MODEL_RERANKER]


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

