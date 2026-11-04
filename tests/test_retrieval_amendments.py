from types import SimpleNamespace

import numpy as np
import pytest

from ragdn.retrieval import Retriever


def _chunk(doc_id, article, text, *, role="chinh", effective=True, part=0):
    return {
        "chunk_id": f"{doc_id}:d{article}:p{part}",
        "doc_id": doc_id,
        "dieu": article,
        "text": text,
        "vai_tro": role,
        "hieu_luc": effective,
    }


def _retriever(chunks, *, vector, bm25, bm25_noacc):
    return Retriever(
        chunks=chunks,
        emb=np.asarray(vector).reshape(-1, 1),
        embedder=SimpleNamespace(encode=lambda query, **kwargs: np.array([1.0])),
        reranker=SimpleNamespace(predict=lambda pairs: np.full(len(pairs), 0.9)),
        bm25=SimpleNamespace(get_scores=lambda tokens: np.asarray(bm25)),
        bm25_noacc=SimpleNamespace(get_scores=lambda tokens: np.asarray(bm25_noacc)),
    )


MAIN = _chunk("luat_dn_2020", "218", "Quy định chuyển tiếp về sở hữu chéo.")
TRANSITION = _chunk(
    "luat_76_2025",
    "3",
    "Doanh nghiệp thành lập trước ngày luật có hiệu lực bổ sung thông tin về chủ sở hữu "
    "hưởng lợi tại lần đăng ký hoặc thông báo thay đổi nội dung đăng ký doanh nghiệp gần nhất.",
    role="sua_doi",
)
REPEALED_MAIN = _chunk("nd_47_2021", "13", "Quy định đã bị bãi bỏ.", effective=False)
REPEALED_AMENDMENT = _chunk(
    "amendment_repealed", "3", "Quy định chuyển tiếp đã bị bãi bỏ.", role="sua_doi", effective=False
)
OTHER_ROLE = _chunk("reference", "3", "Bản tham khảo về quy định chuyển tiếp.", role="tham_khao")


@pytest.mark.parametrize("amendment_channel", ["vector", "bm25", "bm25_noacc"])
def test_transition_can_be_found_through_each_search_channel(amendment_channel):
    chunks = [MAIN, TRANSITION, REPEALED_MAIN, REPEALED_AMENDMENT, OTHER_ROLE]
    # With one candidate per channel, only the selected channel can nominate the amendment.
    main_first = [0.8, 0.4, 0.99, 0.98, 0.97]
    amendment_first = [0.4, 0.8, 0.99, 0.98, 0.97]
    scores = {"vector": main_first, "bm25": main_first, "bm25_noacc": main_first}
    scores[amendment_channel] = amendment_first
    retriever = _retriever(chunks, **scores)

    result = retriever.retrieve(
        "Doanh nghiệp đã thành lập trước ngày luật có hiệu lực phải bổ sung thông tin "
        "chủ sở hữu hưởng lợi khi nào?",
        n_cand=1,
        n_rerank=2,
        top=2,
    )

    assert {chunks[i]["chunk_id"] for i in result["hits"]} == {
        MAIN["chunk_id"],
        TRANSITION["chunk_id"],
    }
    assert result["low_confidence"] is False


def test_explicit_amendment_article_can_be_found_with_low_search_scores():
    same_number = _chunk("luat_dn_2020", "3", "Áp dụng Luật Doanh nghiệp và luật khác.")
    chunks = [MAIN, TRANSITION, same_number]
    scores = [0.8, 0.1, 0.1]
    retriever = _retriever(chunks, vector=scores, bm25=scores, bm25_noacc=scores)

    result = retriever.retrieve("Điều 3 Luật số 76/2025/QH15 quy định gì?", n_cand=1)

    hits = {chunks[i]["chunk_id"] for i in result["hits"]}
    assert TRANSITION["chunk_id"] in hits
    assert same_number["chunk_id"] not in hits
    assert result["flag"] is None


def test_search_can_find_clause_near_end_of_amendment_article():
    earlier = [
        _chunk("luat_76_2025", "1", f"Đoạn sửa đổi trước số {part + 1}.", role="sua_doi", part=part)
        for part in range(8)
    ]
    clause_28 = _chunk(
        "luat_76_2025",
        "1",
        "28. Thay thế từ “sách nhiễu” bằng từ “nhũng nhiễu” tại khoản 1 Điều 16.",
        role="sua_doi",
        part=8,
    )
    chunks = [*earlier, clause_28]
    scores = [0.1] * len(earlier) + [0.9]
    retriever = _retriever(chunks, vector=scores, bm25=scores, bm25_noacc=scores)

    result = retriever.retrieve("Khoản 28 Điều 1 Luật số 76/2025/QH15 sửa cụm từ nào?", n_cand=1)

    assert clause_28["chunk_id"] in {chunks[i]["chunk_id"] for i in result["hits"]}
