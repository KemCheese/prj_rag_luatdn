from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import ragdn.retrieval as retrieval
from ragdn.retrieval import Retriever
from ragdn.pipeline import RAG


def _retriever(labels, scores, *, dense=None, sparse=None):
    chunks = [
        {
            "chunk_id": f"luat_dn_2020:d{i + 1}:p0",
            "doc_id": "luat_dn_2020",
            "dieu": str(i + 1),
            "text": f"Nguồn {i}",
            "loai_hinh": label,
            "hieu_luc": True,
            "vai_tro": "chinh",
        }
        for i, label in enumerate(labels)
    ]
    dense = np.asarray(dense if dense is not None else np.arange(len(chunks), 0, -1), dtype=float)
    sparse = np.asarray(sparse if sparse is not None else dense, dtype=float)
    embedder = SimpleNamespace(encode=Mock(return_value=np.array([1.0])))
    by_text = dict(zip((c["text"] for c in chunks), scores))
    reranker = SimpleNamespace(predict=Mock(side_effect=lambda pairs: np.array([by_text[t] for _, t in pairs])))
    bm25 = SimpleNamespace(get_scores=lambda tokens: sparse)
    retriever = Retriever(chunks, dense.reshape(-1, 1), embedder, reranker, bm25, bm25)
    return retriever


def test_detection_uses_raw_question_once(monkeypatch):
    retriever = _retriever(["tnhh_mtv"], [0.8])
    retriever.glossary = {"tnhh": "trách nhiệm hữu hạn"}
    question = "TNHH MTV có quyền gì?"
    detector = Mock(wraps=retrieval.detect_loai_hinh)
    monkeypatch.setattr(retrieval, "detect_loai_hinh", detector)

    result = retriever.retrieve(question)

    detector.assert_called_once_with(question)
    retriever.embedder.encode.assert_called_once_with(retriever.norm_query(question), normalize_embeddings=True)
    assert result["loai_hinh"] == "tnhh_mtv"


def test_extra_dense_list_can_bring_matching_type_into_rerank():
    retriever = _retriever(["ctcp", "dntn", "tnhh_2tv"], [0.9, 0.8, 0.7], dense=[3, 1, 2], sparse=[3, 2, 1])
    result = retriever.retrieve("TNHH 2TV họp thế nào?", n_cand=2, n_rerank=2, top=2)

    assert 2 in result["hits"]
    pairs = retriever.reranker.predict.call_args.args[0]
    assert any(text == retriever.chunks[2]["text"] for _, text in pairs)


def test_conflict_priority_preserves_raw_scores():
    retriever = _retriever(["ctcp", "tnhh_2tv", None], [0.95, 0.6, 0.7])

    result = retriever.retrieve("TNHH 2TV họp thế nào?")

    assert result["hits"] == [2, 1, 0]
    assert result["max_score"] == 0.95
    assert result["low_confidence"] is False


def test_weak_match_does_not_hide_strong_global_evidence():
    retriever = _retriever(["tnhh_2tv", "ctcp"], [0.1, 0.9])

    result = retriever.retrieve("TNHH 2TV họp thế nào?", top=1)

    assert result["hits"] == [1]
    assert result["max_score"] == 0.9
    assert result["low_confidence"] is False


@pytest.mark.parametrize("question", ["Thủ tục đăng ký doanh nghiệp?", "Chuyển TNHH MTV thành CTCP?"])
@pytest.mark.parametrize("scores", [[0.05, 0.1], [0.15, 0.7]])
def test_unknown_or_multiple_types_preserve_old_result(question, scores):
    retriever = _retriever(["tnhh_mtv", "ctcp"], scores)
    result = retriever.retrieve(question)
    ranked = sorted(range(2), key=lambda i: -scores[i])

    assert result == {
        "hits": [i for i in ranked if scores[i] >= retriever.rerank_min],
        "max_score": max(scores),
        "low_confidence": max(scores) < retriever.rerank_min,
        "flag": None,
        "loai_hinh": None,
    }


@pytest.mark.parametrize(
    ("question", "labels"),
    [
        ("Doanh nghiệp nhà nước có quyền gì?", ["tnhh_mtv", "tnhh_2tv", "ctcp", "dnnn"]),
        ("Nhóm công ty có quyền gì?", ["tnhh_mtv", "ctcp", "hop_danh", "dntn"]),
    ],
)
def test_compatible_types_keep_reranker_order(question, labels):
    retriever = _retriever(labels, [0.9, 0.8, 0.7, 0.6])
    assert retriever.retrieve(question)["hits"] == [0, 1, 2, 3]


@pytest.mark.parametrize("role", ["chinh", "sua_doi"])
@pytest.mark.parametrize("channel", ["dense", "bm25", "bm25_noacc"])
def test_unlabelled_documents_remain_searchable_with_typed_query(role, channel):
    retriever = _retriever(["ctcp", None], [0.6, 0.9], dense=[0.8, 0.4])
    retriever.chunks[1]["doc_id"] = "luat_76_2025" if role == "sua_doi" else "nd_168_2025"
    retriever.chunks[1]["vai_tro"] = role
    # Các fixture cũ không có metadata loại hình vẫn được đọc bằng .get().
    del retriever.chunks[1]["loai_hinh"]
    if channel == "dense":
        retriever.emb = np.array([[0.4], [0.8]])
    elif channel == "bm25":
        retriever.bm25 = SimpleNamespace(get_scores=lambda tokens: np.array([0.4, 0.8]))
    else:
        retriever.bm25_noacc = SimpleNamespace(get_scores=lambda tokens: np.array([0.4, 0.8]))

    assert retriever.retrieve("Công ty cổ phần đăng ký thế nào?", n_cand=1, n_rerank=2, top=1)["hits"] == [1]


def test_explicit_article_survives_type_conflict_and_low_scores():
    retriever = _retriever(["ctcp", "tnhh_mtv"], [0.1, 0.05])
    retriever.chunks[0]["hieu_luc"] = False
    retriever.valid[0] = False

    result = retriever.retrieve("Điều 1 Luật Doanh nghiệp áp dụng cho TNHH MTV thế nào?")

    assert result["hits"] == [0]
    assert result["low_confidence"] is False
    assert result["loai_hinh"] == "tnhh_mtv"


def test_all_below_threshold_retains_raw_score_and_refuses():
    retriever = _retriever(["tnhh_2tv", "ctcp"], [0.1, 0.14])
    result = retriever.retrieve("TNHH 2TV họp thế nào?")

    assert result["hits"] == []
    assert result["max_score"] == 0.14
    assert result["low_confidence"] is True


@pytest.mark.parametrize("minimum", [0.0, 0.15])
def test_no_candidates_does_not_call_reranker(minimum):
    retriever = _retriever(["tnhh_2tv"], [0.9])
    retriever.rerank_min = minimum
    retriever.valid[:] = False
    result = retriever.retrieve("TNHH 2TV họp thế nào?")

    assert result["hits"] == []
    assert result["max_score"] == 0.0
    assert result["low_confidence"] is True
    retriever.reranker.predict.assert_not_called()


@pytest.mark.parametrize("question", ["TNHH 2TV họp thế nào?", "Thủ tục đăng ký doanh nghiệp?"])
def test_zero_top_with_zero_threshold_still_refuses_without_sources(question):
    retriever = _retriever(["tnhh_2tv"], [0.9])
    retriever.rerank_min = 0.0

    result = retriever.retrieve(question, top=0)

    assert result["hits"] == []
    assert result["max_score"] == 0.0
    assert result["low_confidence"] is True


def test_pipeline_forwards_type_without_changing_existing_fields():
    retriever = _retriever(["tnhh_mtv", "ctcp"], [0.8, 0.9])
    rag = RAG({}, retriever)

    result = rag.retrieve("TNHH MTV có quyền gì?")
    expected = {
        "hits": [0, 1],
        "max_score": 0.9,
        "low_confidence": False,
        "flag": None,
        "loai_hinh": "tnhh_mtv",
    }
    assert {key: result[key] for key in expected} == expected
    assert isinstance(result["hits"], list)
    assert result["processing"]["status"] == "not_assessed"
    assert len(result["sources"]) == 2 and len(result["context_spans"]) == 2


def main() -> int:
    return int(pytest.main([__file__, *sys.argv[1:]]))


if __name__ == "__main__":
    raise SystemExit(main())
