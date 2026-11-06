from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (str(ROOT), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest

import ragdn.generate as gen
from ragdn.loai_hinh import LABELS
from ragdn.manifest import load_manifest
from ragdn.config import MANIFEST_PATH
from ragdn.query_models import (
    EvidenceRecord,
    IssueAssessment,
    IssueRecord,
    IssueScope,
    ProcessingReport,
    QuestionPlan,
    SourceSpan,
)
from ragdn.query_pipeline import QueryResult


def _source(dieu="59", loai_hinh="tnhh_2tv", *, doc_id="luat_dn_2020", notes=None):
    return {
        "doc_id": doc_id, "dieu": dieu, "loai_hinh": loai_hinh,
        "ten_vb": gen._documents()[doc_id].ten_ngan,
        "vai_tro": "sua_doi" if doc_id == "luat_76_2025" else "chinh",
        "khoan": ["1"], "hieu_luc": True, "sua_doi": notes or [], "refs": [],
        "text": f"Điều {dieu}. Quy định\n1. Nội dung quy định.",
    }


def _retriever(sources, q_lh="tnhh_2tv"):
    return SimpleNamespace(
        chunks=sources,
        retrieve=lambda q: {"hits": list(range(len(sources))), "low_confidence": False,
                            "flag": None, "loai_hinh": q_lh},
    )


def test_context_labels_conflicts_and_compatible_sources():
    sources = [_source(), _source("148", "ctcp"), _source("1", None)]
    context = gen.build_context(sources, "tnhh_2tv")
    assert f"[Áp dụng cho: {LABELS['tnhh_2tv']}]" in context
    assert f"[KHÔNG áp dụng cho {LABELS['tnhh_2tv']}]" in context
    assert "Áp dụng cho" not in gen.build_context([sources[2]], "tnhh_2tv")
    assert "KHÔNG áp dụng" not in gen.build_context([sources[0]], "dnnn")
    assert "KHÔNG áp dụng" not in gen.build_context(sources, None)


def test_t9_label_is_not_repeated_in_context():
    source = _source()
    source["text"] = f"Luật Doanh nghiệp | Chương III | Mục 1 ({LABELS['tnhh_2tv']}) | Điều 59. Quy định\n1. Nội dung."
    assert gen.build_context([source], "tnhh_2tv").count(LABELS["tnhh_2tv"]) == 1


def test_amendment_relation_survives_manifest_and_is_displayed():
    doc = next(d for d in load_manifest(MANIFEST_PATH).documents if d.doc_id == "luat_76_2025")
    assert doc.sua_doi_cho == "luat_dn_2020"
    source = _source("3", None, doc_id=doc.doc_id)
    assert "[Văn bản sửa đổi Luật Doanh nghiệp]" in gen.build_context([source])


@pytest.mark.parametrize("marker", gen._LEAK_MARKERS)
def test_prompt_leak_is_detected_and_pruned(marker):
    source = _source()
    answer = f"Nội dung quy định tại Điều 59 [1].\n{marker} [1]."
    assert any("hướng dẫn nội bộ" in issue for issue in gen.find_issues(answer, [source]))
    text, count = gen.salvage(answer, [source])
    assert marker not in text
    assert count == 1


def test_valid_missing_information_phrase_is_allowed():
    answer = "Nội dung quy định tại Điều 59 [1]. Các văn bản đã nạp chưa quy định nội dung này."
    assert gen.find_issues(answer, [_source()]) == []


def test_generate_uses_retrieval_type_in_prompt(monkeypatch):
    captured = []
    monkeypatch.setattr(gen, "call_llm", lambda system, user: captured.append(user) or "Nội dung quy định tại Điều 59 [1].")
    result = gen.generate_answer("Quy định là gì?", _retriever([_source()]))
    assert f"<loai_hinh>{LABELS['tnhh_2tv']}</loai_hinh>" in captured[0]
    assert result["partial"] is False


def test_type_validator_uses_cited_source_not_article_number():
    sources = [_source("59", "ctcp"), _source("59", "tnhh_2tv", doc_id="nd_168_2025")]
    assert any("nguồn [1]" in issue for issue in gen.find_issues("Nội dung Điều 59 [1].", sources, q_lh="tnhh_2tv"))
    assert gen.find_issues("Nội dung Điều 59 [2].", sources, q_lh="tnhh_2tv") == []
    assert gen.find_issues("Nội dung Điều 59 [1].", sources) == []


@pytest.mark.parametrize(
    "answer",
    [
        "Điều 148 [1] không áp dụng cho công ty TNHH hai thành viên trở lên.",
        "Điều 148 không được áp dụng cho công ty TNHH hai thành viên trở lên [1].",
        "Điều 148 chỉ áp dụng cho Công ty cổ phần [1].",
    ],
)
def test_valid_explanation_of_non_applicability_is_kept(answer):
    assert gen.find_issues(answer, [_source("148", "ctcp")], q_lh="tnhh_2tv") == []


@pytest.mark.parametrize("connector", ["nhưng", "tuy nhiên", "đồng thời", ";", ",", "và", "mà"])
def test_negative_clause_does_not_exempt_later_assertion(connector):
    answer = f"Điều 148 [1] không áp dụng cho TNHH hai thành viên {connector} nghị quyết vẫn được thông qua [1]."
    assert gen.find_issues(answer, [_source("148", "ctcp")], q_lh="tnhh_2tv")


def test_only_applies_is_not_an_unconditional_bypass():
    answer = "Điều 148 chỉ áp dụng cho Công ty TNHH hai thành viên trở lên [1]."
    assert gen.find_issues(answer, [_source("148", "ctcp")], q_lh="tnhh_2tv")


@pytest.mark.parametrize("answer", [
    "Điều 148 chỉ áp dụng cho Công ty cổ phần và Công ty TNHH hai thành viên vẫn được thông qua nghị quyết [1].",
    "Điều 148 không thuộc hồ sơ đăng ký, nghị quyết được thông qua theo Điều 148 [1].",
])
def test_unrelated_negative_or_extra_claim_does_not_bypass_type_check(answer):
    assert gen.find_issues(answer, [_source("148", "ctcp")], q_lh="tnhh_2tv")


@pytest.mark.parametrize("answer", [
    "Điều 148 [1] không áp dụng cho TNHH hai thành viên, nhưng công ty vẫn thông qua theo Điều 148.",
    "Điều 148 không áp dụng cho TNHH hai thành viên và công ty vẫn áp dụng Điều 148 [1].",
])
def test_uncited_assertion_does_not_inherit_negative_exception(answer):
    assert gen.find_issues(answer, [_source("148", "ctcp")], q_lh="tnhh_2tv")


def test_citation_in_lead_in_is_not_treated_as_applicability_claim():
    answer = "Theo Điều 148 [1], quy định này không áp dụng cho công ty TNHH hai thành viên."
    assert gen.find_issues(answer, [_source("148", "ctcp")], q_lh="tnhh_2tv") == []


def test_contextual_lead_in_before_non_applicability_is_kept():
    answer = "Đối với công ty TNHH hai thành viên, Điều 148 không áp dụng [1]."
    assert gen.find_issues(answer, [_source("148", "ctcp")], q_lh="tnhh_2tv") == []


def test_uncited_initial_assertion_is_still_checked():
    answer = "Công ty vẫn thông qua nghị quyết, Điều 148 không áp dụng [1]."
    assert gen.find_issues(answer, [_source("148", "ctcp")], q_lh="tnhh_2tv")


def test_comma_after_article_number_still_separates_claims():
    answer = "Không áp dụng Điều 148, nghị quyết vẫn được thông qua [1]."
    assert gen.find_issues(answer, [_source("148", "ctcp")], q_lh="tnhh_2tv")


def test_decimal_comma_is_not_a_clause_boundary():
    source = _source("148", "ctcp")
    source["text"] += " Tỷ lệ 65,5%."
    answer = "Điều 148 không áp dụng cho TNHH hai thành viên với tỷ lệ 65,5% [1]."
    assert gen.find_issues(answer, [source], q_lh="tnhh_2tv") == []


def test_negative_clause_does_not_bypass_article_or_numeric_validation():
    answer = "Điều 999 không áp dụng trong 999 ngày [1]."
    issues = gen.find_issues(answer, [_source("148", "ctcp")], q_lh="tnhh_2tv")
    assert any("Điều" in issue for issue in issues)
    assert any("số không có" in issue for issue in issues)


@pytest.mark.parametrize("q_lh", ["dnnn", "nhom_cty", None])
def test_compatible_or_unspecified_question_type_accepts_source(q_lh):
    assert gen.find_issues("Nội dung Điều 59 [1].", [_source()], q_lh=q_lh) == []


def test_unlabelled_source_is_not_rejected_for_company_type():
    assert gen.find_issues("Nội dung Điều 3 [1].", [_source("3", None, doc_id="luat_76_2025")], q_lh="tnhh_2tv") == []


@pytest.mark.parametrize(
    ("source", "answer"),
    [
        (_source("3", None), "Điều 3 Luật số 76/2025/QH15 quy định việc áp dụng luật [1]."),
        (
            {**_source("1", None, doc_id="luat_76_2025"), "refs": ["128"]},
            "Điều 128 Luật số 76/2025/QH15 quy định về chào bán trái phiếu [1].",
        ),
        (_source("3", None), "Luật số 76/2025/QH15 có quy định liên quan [1]."),
    ],
)
def test_document_identity_cannot_be_borrowed_from_other_corpus_sources(source, answer):
    assert gen.find_issues(answer, [source])


def test_amendment_can_reference_target_article():
    source = _source("1", None, doc_id="luat_76_2025")
    source["refs"] = ["128"]
    answer = "Điều 1 Luật số 76/2025/QH15 sửa Điều 128 Luật Doanh nghiệp [1]."
    assert gen.find_issues(answer, [source]) == []


def test_amendment_mentions_share_trailing_citation_in_multi_source_sentence():
    amendment = _source("1", None, doc_id="luat_76_2025")
    amendment["refs"] = ["128"]
    sources = [_source("59", None), amendment]
    answer = (
        "Điều 59 Luật Doanh nghiệp quy định cuộc họp [1], còn Điều 1 Luật số 76/2025/QH15 "
        "sửa đổi Điều 128 Luật Doanh nghiệp [2]."
    )
    assert gen.find_issues(answer, sources) == []
    assert gen.find_issues(answer.replace("Điều 128 Luật Doanh nghiệp", "Điều 128 Luật số 76/2025/QH15"), sources)


def test_document_spans_do_not_bridge_two_article_mentions():
    sources = [_source("59", None), _source("3", None, doc_id="luat_76_2025")]
    answer = "Điều 59 Luật Doanh nghiệp [1], Điều 3 Luật số 76/2025/QH15 [2] quy định nội dung."
    assert gen.find_issues(answer, sources) == []
    amendment = _source("1", None, doc_id="luat_76_2025")
    amendment["refs"] = ["128"]
    assert gen.find_issues(
        "Theo Điều 1 Luật số 76/2025/QH15, Điều 128 Luật Doanh nghiệp được sửa đổi [1].", [amendment]
    ) == []


def test_full_amendment_title_does_not_inherit_target_document_identity():
    source = _source("2", None, doc_id="luat_76_2025")
    title = gen._documents()["luat_76_2025"].ten_vb
    assert gen.find_issues(f"Theo {title}, Điều 2 quy định nội dung [1].", [source]) == []


def test_same_number_document_citations_are_checked_locally():
    sources = [_source("3", None), _source("3", None, doc_id="luat_76_2025")]
    correct = "Điều 3 Luật Doanh nghiệp [1] và Điều 3 Luật số 76/2025/QH15 [2]."
    wrong = "Điều 3 Luật số 76/2025/QH15 [1] và Điều 3 Luật Doanh nghiệp [2]."
    assert gen.find_issues(correct, sources) == []
    assert gen.find_issues(wrong, sources)


def test_embedded_citation_cannot_borrow_later_source_identity():
    sources = [_source("3", None), _source("3", None, doc_id="luat_76_2025")]
    answer = "Điều 3 [1] Luật số 76/2025/QH15 và Điều 3 Luật số 76/2025/QH15 [2] quy định nội dung."
    assert gen.find_issues(answer, sources)


def test_prefix_citations_belong_to_their_document_mentions():
    sources = [_source("3", None), _source("3", None, doc_id="luat_76_2025")]
    correct = "[1] Luật Doanh nghiệp, Điều 3 và [2] Luật số 76/2025/QH15, Điều 3 quy định nội dung."
    wrong = "[1] Luật số 76/2025/QH15, Điều 3 và [2] Luật Doanh nghiệp, Điều 3 quy định nội dung."
    assert gen.find_issues(correct, sources) == []
    assert gen.find_issues(wrong, sources)


def test_document_before_article_is_validated():
    source = _source("4", None, notes=["Điều 1 Luật số 76/2025/QH15 sửa đổi nội dung."])
    assert gen.find_issues("Theo Luật số 76/2025/QH15, Điều 4 quy định nội dung [1].", [source])
    assert gen.find_issues("Theo Luật Doanh nghiệp, Điều 4 quy định nội dung [1].", [source]) == []
    assert gen.find_issues("Theo Luật số 76/2025/QH15, khoản 1 Điều 4 quy định nội dung [1].", [source])
    assert gen.find_issues("Theo Luật Doanh nghiệp, khoản 1 Điều 4 quy định nội dung [1].", [source]) == []


def test_uncited_mention_cannot_borrow_other_individual_citations():
    sources = [_source("3", None), _source("3", None, doc_id="luat_76_2025")]
    answer = "Điều 3 Luật số 76/2025/QH15, [1] Điều 3 Luật Doanh nghiệp và [2] Điều 3 Luật số 76/2025/QH15."
    assert gen.find_issues(answer, sources)
    assert gen.find_issues("Điều 3 Luật Doanh nghiệp và Điều 3 Luật số 76/2025/QH15 [1][2].", sources) == []


def test_unqualified_reference_survives_same_number_in_amendment_note():
    source = _source("4", None, notes=["Điều 1 Luật số 76/2025/QH15 sửa đổi nội dung."])
    source["text"] += " Theo Điều 1."
    source["refs"] = ["1"]
    assert gen.find_issues("Điều 1 Luật Doanh nghiệp quy định nội dung [1].", [source]) == []


def test_type_is_used_through_retry_and_salvage(monkeypatch):
    sources = [_source(), _source("148", "ctcp")]
    bad = "Nội dung Điều 59 [1].\nNghị quyết theo Điều 148 [2]."
    calls = []

    def fake_call(system, user, **kwargs):
        calls.append(user)
        return bad

    monkeypatch.setattr(gen, "call_llm", fake_call)
    result = gen.generate_answer("Công ty TNHH hai thành viên có quyền gì?", _retriever(sources))
    assert len(calls) == 2
    assert result["partial"] is True
    assert "Điều 59" in result["answer"]
    assert "Điều 148" not in result["answer"]


def test_salvage_drops_only_wrong_type_sentences():
    sources = [_source(), _source("148", "ctcp")]
    answer = "Nội dung Điều 59 [1].\nNghị quyết theo Điều 148 [2]."
    text, count = gen.salvage(answer, sources, q_lh="tnhh_2tv")
    assert count == 1
    assert text == "Nội dung Điều 59 [1]."


def test_multi_issue_distinct_company_types_do_not_impose_whole_question_company_type(monkeypatch):
    s_tnhh = {**_source("59", "tnhh_2tv"), "chunk_id": "luat_dn_2020:d59:p0"}
    s_ctcp = {**_source("148", "ctcp"), "chunk_id": "luat_dn_2020:d148:p0"}
    plan = QuestionPlan(
        original_question="So sánh TNHH 2TV và CTCP?",
        facts=[],
        issues=[
            IssueRecord(issue_id="i1", title="Quy định TNHH 2TV", main_query="tnhh 2tv", scope=IssueScope(company_types=["tnhh_2tv"])),
            IssueRecord(issue_id="i2", title="Quy định CTCP", main_query="ctcp", scope=IssueScope(company_types=["ctcp"])),
        ],
    )
    ev1 = EvidenceRecord(
        evidence_id="e1",
        issue_id="i1",
        parent_chunk_id=s_tnhh["chunk_id"],
        doc_id=s_tnhh["doc_id"],
        dieu=s_tnhh["dieu"],
        source_spans=[SourceSpan(start=0, end=len(s_tnhh["text"]), quote=s_tnhh["text"])],
    )
    ev2 = EvidenceRecord(
        evidence_id="e2",
        issue_id="i2",
        parent_chunk_id=s_ctcp["chunk_id"],
        doc_id=s_ctcp["doc_id"],
        dieu=s_ctcp["dieu"],
        source_spans=[SourceSpan(start=0, end=len(s_ctcp["text"]), quote=s_ctcp["text"])],
    )
    qres = QueryResult(
        original_question="So sánh TNHH 2TV và CTCP?",
        retrieval={"hits": [0, 1], "max_score": 0.9, "low_confidence": False, "flag": "query_pipeline", "loai_hinh": "tnhh_2tv"},
        sources=(s_tnhh, s_ctcp),
        context_views=(),
        plan=plan,
        evidence=(ev1, ev2),
        processing=ProcessingReport(
            mode="enhanced",
            route_reason="multi_issue_question",
            status="supported",
            issues=[
                IssueAssessment(issue_id="i1", status="supported", evidence_ids=["e1"]),
                IssueAssessment(issue_id="i2", status="supported", evidence_ids=["e2"]),
            ],
        ),
        diagnostics={},
    )
    ans_both = "Theo Điều 59 Luật Doanh nghiệp, nội dung quy định [1].\nTheo Điều 148 Luật Doanh nghiệp, nội dung quy định [2]."
    monkeypatch.setattr(gen, "call_llm", lambda system, user, **kw: ans_both)
    res = gen.generate_answer("So sánh TNHH 2TV và CTCP?", _retriever([s_tnhh, s_ctcp]), query_result=qres)
    assert res["partial"] is False
    assert "Điều 59" in res["answer"] and "Điều 148" in res["answer"]


def main() -> int:
    return int(pytest.main([__file__, *sys.argv[1:]]))


if __name__ == "__main__":
    raise SystemExit(main())
