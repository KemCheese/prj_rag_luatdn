"""Offline P4-A evidence mapping, provenance validation and coverage aggregation tests."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import re
from pydantic import ValidationError
import pytest

from ragdn.article_references import ArticleReference
from ragdn.evidence import (
    Coverage,
    EvidenceCatalog,
    EvidenceProposal,
    IssueMapping,
    MapperOutput,
    MissingProposal,
    Rejection,
    SuppliedGroup,
    article,
    coverage_status,
    document_metadata,
    fold,
    group_error,
    merge_coverage,
    validate_mapping,
)
from ragdn.query_models import (
    FactRecord,
    IssueAssessment,
    IssueRecord,
    IssueScope,
    IssueStatus,
    NotProcessedIssue,
    PartialReason,
    ProcessingStatus,
    QuestionPlan,
)
from ragdn.source_windows import SourceUnit, SourceWindow, WindowGroup, build_source_windows, source_groups


class FakeTokenizer:
    def __call__(self, text, pair=None, *, add_special_tokens=False,
                 truncation=False, return_offsets_mapping=False):
        assert truncation is False
        spans = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        ids = list(range(len(spans)))
        if pair is not None:
            ids += list(range(len(re.findall(r"\S+", pair))))
        if add_special_tokens:
            ids += [999, 999, 999]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = spans
        return result


def make_parent(
    chunk_id: str = "c1",
    text: str = (
        "Trừ trường hợp Điều lệ công ty có quy định khác:\n"
        "1. Hội đồng thành viên gồm tất cả thành viên công ty là cá nhân và người đại diện theo ủy quyền.\n"
        "2. Trừ trường hợp quy định tại khoản 1 Điều này, cuộc họp được tiến hành khi có đủ thành viên."
    ),
    *,
    doc_id: str = "luat_dn_2020",
    dieu: str = "58",
    hieu_luc: bool = True,
    vai_tro: str = "chinh",
    loai_hinh: str | None = "tnhh_2tv",
    sua_doi_cho: str | None = None,
    ngay_hieu_luc: str | None = "2021-01-01",
    **extra,
) -> dict:
    parent = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "dieu": dieu,
        "text": text,
        "hieu_luc": hieu_luc,
        "vai_tro": vai_tro,
        "loai_hinh": loai_hinh,
        "sua_doi_cho": sua_doi_cho,
        "ngay_hieu_luc": ngay_hieu_luc,
    }
    parent.update(extra)
    return parent


def make_supplied(parent: dict, issue_ids: tuple[str, ...] = ("i1",), *, passage_tokens: int = 384) -> tuple[SuppliedGroup, ...]:
    tokenizer = FakeTokenizer()
    windows = build_source_windows(parent, tokenizer, passage_tokens=passage_tokens, overlap_tokens=8)
    groups = source_groups(parent, windows)
    by_id = {window.window_id: window for window in windows}
    return tuple(
        SuppliedGroup(group, tuple(by_id[wid] for wid in group.window_ids), issue_ids)
        for group in groups
    )


def make_plan(
    question: str = "Công ty TNHH hai thành viên hỏi về cuộc họp Hội đồng thành viên?",
    *,
    facts: list[FactRecord] | None = None,
    issues: list[IssueRecord] | None = None,
    references: list[ArticleReference] | None = None,
    not_processed: list[NotProcessedIssue] | None = None,
) -> QuestionPlan:
    fact_list = facts or [FactRecord(fact_id="f1", quote=question, start=0, end=len(question))]
    issue_list = issues or [
        IssueRecord(
            issue_id="i1",
            title="Cuộc họp Hội đồng thành viên",
            fact_ids=[fact_list[0].fact_id],
            scope=IssueScope(company_types=["tnhh_2tv"], doc_ids=["luat_dn_2020"]),
            main_query="cuộc họp Hội đồng thành viên công ty TNHH hai thành viên",
        )
    ]
    return QuestionPlan(
        original_question=question,
        facts=fact_list,
        issues=issue_list,
        references=references or [],
        not_processed=not_processed or [],
    )


def make_catalog(
    parents: list[dict],
    supplied: list[SuppliedGroup] | tuple[SuppliedGroup, ...],
    manifest=None,
) -> EvidenceCatalog:
    return EvidenceCatalog(
        parents={parent["chunk_id"]: parent for parent in parents},
        groups={item.group.group_id: item for item in supplied},
        documents=document_metadata(manifest),
    )


def test_mapper_schemas_forbid_extra_fields_and_derive_exact_parent_spans():
    with pytest.raises(ValidationError):
        EvidenceProposal(
            evidence_id="e1", issue_id="i1", group_id="c1:g0", unit_ids=["c1:u0"],
            doc_id="luat_dn_2020", dieu="58", start=0,
        )
    with pytest.raises(ValidationError):
        MissingProposal(description="Thiếu Điều lệ", extra_field="forbidden")
    with pytest.raises(ValidationError):
        IssueMapping(issue_id="i1", status="supported", unknown="forbidden")
    with pytest.raises(ValidationError):
        MapperOutput(issues=[], extra_root=True)

    parent = make_parent(dieu="17đ")
    supplied = make_supplied(parent)
    plan = make_plan()
    output = MapperOutput(
        evidence=[
            EvidenceProposal(
                evidence_id="e1",
                issue_id="i1",
                group_id=supplied[0].group.group_id,
                unit_ids=[unit.unit_id for unit in supplied[0].group.units],
                doc_id="luat_dn_2020",
                dieu="17dd",
                khoan="1",
                condition_quotes=["Trừ trường hợp Điều lệ công ty có quy định khác"],
            )
        ],
        issues=[IssueMapping(issue_id="i1", status="supported", evidence_ids=["e1"])],
    )
    coverage = validate_mapping(plan, output, make_catalog([parent], supplied), id_prefix="m1:")
    assert coverage.rejections == ()
    assert len(coverage.evidence) == 1
    record = coverage.evidence[0]
    assert record.evidence_id == "m1:e1"
    assert record.dieu == "17đ"
    assert record.khoan == "1"
    assert [span.quote for span in record.source_spans] == [unit.text for unit in supplied[0].group.units]
    for span in record.source_spans:
        assert parent["text"][span.start:span.end] == span.quote


def test_group_error_rejects_missing_parent_non_atomic_duplicate_and_corrupt_spans():
    parent = make_parent()
    valid = make_supplied(parent)[0]
    assert group_error(valid, {"c1": parent}) is None
    assert group_error(valid, {}) == "source_parent_missing"
    assert group_error(valid, {"c1": {**parent, "text": ""}}) == "source_parent_missing"

    non_atomic = SuppliedGroup(
        WindowGroup("c1:g0", "c1", valid.group.units, valid.group.window_ids, atomic=False),
        valid.windows,
        ("i1",),
    )
    assert group_error(non_atomic, {"c1": parent}) == "source_group_not_atomic"

    u0 = valid.group.units[0]
    dup_units = SuppliedGroup(
        WindowGroup("c1:g0", "c1", (u0, u0), valid.group.window_ids),
        valid.windows,
        ("i1",),
    )
    assert group_error(dup_units, {"c1": parent}) == "source_unit_duplicate"

    bad_unit = SourceUnit("c1:u0", u0.start, u0.end, "Tampered text", u0.khoan)
    corrupt_unit = SuppliedGroup(
        WindowGroup("c1:g0", "c1", (bad_unit, *valid.group.units[1:]), valid.group.window_ids),
        valid.windows,
        ("i1",),
    )
    assert group_error(corrupt_unit, {"c1": parent}) == "source_unit_span_invalid"

    reversed_units = SuppliedGroup(
        WindowGroup("c1:g0", "c1", (valid.group.units[1], valid.group.units[0]), valid.group.window_ids),
        valid.windows,
        ("i1",),
    )
    assert group_error(reversed_units, {"c1": parent}) == "source_unit_span_invalid"

    missing_win = SuppliedGroup(valid.group, (), ("i1",))
    assert group_error(missing_win, {"c1": parent}) == "source_window_missing"

    bad_win = SourceWindow(valid.windows[0].window_id, "c1", 0, 5, "wrong", 1)
    corrupt_win = SuppliedGroup(valid.group, (bad_win,), ("i1",))
    assert group_error(corrupt_win, {"c1": parent}) == "source_window_span_invalid"


def test_repeated_clause_whole_parent_atomic_group_stays_intact():
    repeated_text = (
        "Sửa đổi, bổ sung một số khoản của Điều 58 như sau:\n"
        "1. Hội đồng thành viên họp ít nhất mỗi năm một lần theo quy định của Điều lệ.\n"
        "1. Hội đồng thành viên có quyền quyết định chiến lược phát triển của công ty.\n"
        "2. Nghị quyết Hội đồng thành viên được thông qua khi đủ tỷ lệ biểu quyết."
    )
    parent = make_parent(chunk_id="c_rep", text=repeated_text)
    supplied = make_supplied(parent, passage_tokens=20)
    assert len(supplied) == 1
    whole_group = supplied[0].group
    assert len(whole_group.units) == 4
    assert len(whole_group.window_ids) >= 2

    partial_supplied = SuppliedGroup(
        WindowGroup("c_rep:g0", "c_rep", whole_group.units[:2], whole_group.window_ids),
        supplied[0].windows,
        ("i1",),
    )
    assert group_error(partial_supplied, {"c_rep": parent}) == "source_group_not_atomic"

    plan = make_plan()
    catalog = make_catalog([parent], supplied)
    incomplete_units = MapperOutput(
        evidence=[
            EvidenceProposal(
                evidence_id="e_sub", issue_id="i1", group_id=whole_group.group_id,
                unit_ids=[whole_group.units[1].unit_id], doc_id="luat_dn_2020", dieu="58",
            ),
            EvidenceProposal(
                evidence_id="e_win", issue_id="i1", group_id=whole_group.group_id,
                unit_ids=[u.unit_id for u in whole_group.units],
                window_ids=[whole_group.window_ids[0]],
                doc_id="luat_dn_2020", dieu="58",
            ),
            EvidenceProposal(
                evidence_id="e_amb_khoan", issue_id="i1", group_id=whole_group.group_id,
                unit_ids=[u.unit_id for u in whole_group.units],
                doc_id="luat_dn_2020", dieu="58", khoan="1",
            ),
            EvidenceProposal(
                evidence_id="e_valid", issue_id="i1", group_id=whole_group.group_id,
                unit_ids=[u.unit_id for u in whole_group.units],
                window_ids=list(whole_group.window_ids),
                doc_id="luat_dn_2020", dieu="58", khoan="2",
            ),
        ],
        issues=[IssueMapping(issue_id="i1", status="supported", evidence_ids=["e_valid"])],
    )
    coverage = validate_mapping(plan, incomplete_units, catalog)
    assert [(r.proposal_id, r.reason) for r in coverage.rejections] == [
        ("e_sub", "evidence_atomic_group_incomplete"),
        ("e_win", "evidence_atomic_group_incomplete"),
        ("e_amb_khoan", "evidence_clause_not_recognizable"),
    ]
    assert len(coverage.evidence) == 1
    assert len(coverage.evidence[0].source_spans) == 4
    assert coverage.assessments[0].status == IssueStatus.UNCERTAIN
    assert "mapping_evidence_rejected" in coverage.assessments[0].reasons


def test_evidence_rejection_for_ids_doc_article_clause_point_and_non_substantive():
    text = (
        "1. Hội đồng thành viên có các quyền và nghĩa vụ theo quy định của Luật này:\n"
        "a) Quyết định chiến lược phát triển và kế hoạch kinh doanh hằng năm của công ty;\n"
        "2. Trừ trường hợp quy định tại khoản 1 Điều này, thành viên có quyền biểu quyết."
    )
    parent = make_parent(chunk_id="c1", text=text, dieu="17d")
    header_parent = make_parent(
        chunk_id="c_hdr",
        text="Chương III\nĐiều 58. Hội đồng thành viên\n1. Xem Điều 59 và Điều 60 Luật Doanh nghiệp.\n[Tham chiếu: Điều 61]",
    )
    s1 = make_supplied(parent, issue_ids=("i1",))[0]
    s_hdr = make_supplied(header_parent, issue_ids=("i1",))[0]
    plan = make_plan()
    catalog = make_catalog([parent, header_parent], [s1, s_hdr])
    units = [u.unit_id for u in s1.group.units]

    output = MapperOutput(
        evidence=[
            EvidenceProposal(evidence_id="dup", issue_id="i1", group_id=s1.group.group_id, unit_ids=units, doc_id="luat_dn_2020", dieu="17d"),
            EvidenceProposal(evidence_id="dup", issue_id="i1", group_id=s1.group.group_id, unit_ids=units, doc_id="luat_dn_2020", dieu="17d"),
            EvidenceProposal(evidence_id="unk_issue", issue_id="i9", group_id=s1.group.group_id, unit_ids=units, doc_id="luat_dn_2020", dieu="17d"),
            EvidenceProposal(evidence_id="unk_grp", issue_id="i1", group_id="missing:g0", unit_ids=units, doc_id="luat_dn_2020", dieu="17d"),
            EvidenceProposal(evidence_id="bad_win", issue_id="i1", group_id=s1.group.group_id, unit_ids=units, window_ids=["unknown_w"], doc_id="luat_dn_2020", dieu="17d"),
            EvidenceProposal(evidence_id="bad_suffix", issue_id="i1", group_id=s1.group.group_id, unit_ids=units, doc_id="luat_dn_2020", dieu="17đ"),
            EvidenceProposal(evidence_id="bad_khoan", issue_id="i1", group_id=s1.group.group_id, unit_ids=units, doc_id="luat_dn_2020", dieu="17d", khoan="9"),
            EvidenceProposal(evidence_id="wrong_clause_diem", issue_id="i1", group_id=s1.group.group_id, unit_ids=units, doc_id="luat_dn_2020", dieu="17d", khoan="2", diem="a"),
            EvidenceProposal(evidence_id="hdr_only", issue_id="i1", group_id=s_hdr.group.group_id, unit_ids=[u.unit_id for u in s_hdr.group.units], doc_id="luat_dn_2020", dieu="58"),
            EvidenceProposal(evidence_id="bad_cond", issue_id="i1", group_id=s1.group.group_id, unit_ids=units, doc_id="luat_dn_2020", dieu="17d", condition_quotes=["Điều kiện không có trong nguồn"]),
            EvidenceProposal(evidence_id="ok", issue_id="i1", group_id=s1.group.group_id, unit_ids=units, doc_id="luat_dn_2020", dieu="17d", khoan="1", diem="a"),
        ],
        issues=[
            IssueMapping(issue_id="i1", status="supported", evidence_ids=["ok"]),
            IssueMapping(issue_id="i_unknown", status="supported", evidence_ids=[]),
        ],
    )
    coverage = validate_mapping(plan, output, catalog)
    reasons = {(r.proposal_id, r.reason) for r in coverage.rejections}
    assert ("dup", "evidence_duplicate_id") in reasons
    assert ("unk_issue", "evidence_unknown_issue") in reasons
    assert ("unk_grp", "evidence_group_not_supplied") in reasons
    assert ("bad_win", "evidence_window_not_supplied") in reasons
    assert ("bad_suffix", "evidence_document_article_mismatch") in reasons
    assert ("bad_khoan", "evidence_clause_not_recognizable") in reasons
    assert ("wrong_clause_diem", "evidence_point_not_recognizable") in reasons
    assert ("hdr_only", "evidence_not_substantive") in reasons
    assert ("bad_cond", "evidence_condition_not_present") in reasons
    assert ("i_unknown", "mapping_unknown_issue") in reasons
    assert len(coverage.evidence) == 1
    assert coverage.evidence[0].evidence_id == "ok"


def test_role_amendment_scope_company_time_and_repeal_validation():
    p_meta_mismatch = make_parent(chunk_id="c_meta", vai_tro="chinh")
    p_bad_role = make_parent(chunk_id="c_role", doc_id="doc_ref", vai_tro="tham_khao")
    p_no_target = make_parent(chunk_id="c_notgt", doc_id="doc_amend_bad", vai_tro="sua_doi", sua_doi_cho=None)
    p_wrong_target = make_parent(chunk_id="c_wtgt", doc_id="doc_amend_other", vai_tro="sua_doi", sua_doi_cho="luat_dau_tu")
    p_valid_amend = make_parent(chunk_id="c_amend_ok", doc_id="luat_76_2025", vai_tro="sua_doi", sua_doi_cho="luat_dn_2020", ngay_hieu_luc="2025-07-01")
    p_wrong_type = make_parent(chunk_id="c_ctcp", loai_hinh="ctcp")
    p_repealed = make_parent(chunk_id="c_rep", hieu_luc=False, trang_thai="het_hieu_luc")

    parents = [p_meta_mismatch, p_bad_role, p_no_target, p_wrong_target, p_valid_amend, p_wrong_type, p_repealed]
    supplied = [make_supplied(p, issue_ids=("i_cur", "i_hist", "i_comp"))[0] for p in parents]
    manifest = [
        {"doc_id": "luat_dn_2020", "vai_tro": "chinh", "trang_thai": "con_hieu_luc", "ngay_hieu_luc": "2021-01-01"},
        {"doc_id": "luat_76_2025", "vai_tro": "sua_doi", "sua_doi_cho": "luat_dn_2020", "trang_thai": "con_hieu_luc", "ngay_hieu_luc": "2025-07-01"},
    ]
    question = "Ngày 01/03/2025 công ty TNHH 2TV họp; lịch sử điều khoản bị bãi bỏ và so sánh CTCP với TNHH 2TV?"
    f1 = FactRecord(fact_id="f1", quote=question, start=0, end=len(question))
    plan = make_plan(
        question=question,
        facts=[f1],
        issues=[
            IssueRecord(
                issue_id="i_cur", title="Áp dụng ngày 01/03/2025", fact_ids=["f1"],
                scope=IssueScope(company_types=["tnhh_2tv"], doc_ids=["luat_dn_2020"], time_quote="01/03/2025"),
                main_query="họp công ty TNHH hai thành viên ngày 01/03/2025",
            ),
            IssueRecord(
                issue_id="i_hist", title="Quy định bị bãi bỏ trước khi sửa đổi", fact_ids=["f1"],
                scope=IssueScope(company_types=["tnhh_2tv"], doc_ids=["luat_dn_2020"]),
                main_query="quy định bị bãi bỏ",
            ),
            IssueRecord(
                issue_id="i_comp", title="So sánh CTCP và TNHH 2TV", fact_ids=["f1"],
                scope=IssueScope(company_types=["ctcp", "tnhh_2tv"], doc_ids=["luat_dn_2020"]),
                main_query="so sánh CTCP và TNHH hai thành viên",
            ),
        ],
    )
    catalog = make_catalog(parents, supplied, manifest=manifest)
    catalog.documents["luat_dn_2020_conflict"] = {"doc_id": "luat_dn_2020_conflict", "vai_tro": "sua_doi"}
    p_meta_mismatch["doc_id"] = "luat_dn_2020_conflict"

    def prop(eid, iid, p, **kw):
        grp = f"{p['chunk_id']}:g0"
        u_ids = [u.unit_id for u in catalog.groups[grp].group.units]
        return EvidenceProposal(evidence_id=eid, issue_id=iid, group_id=grp, unit_ids=u_ids, doc_id=p["doc_id"], dieu=p["dieu"], **kw)

    output = MapperOutput(
        evidence=[
            prop("e_meta", "i_cur", p_meta_mismatch),
            prop("e_role", "i_cur", p_bad_role),
            prop("e_notgt", "i_cur", p_no_target),
            prop("e_wtgt", "i_cur", p_wrong_target),
            prop("e_future", "i_cur", p_valid_amend),
            prop("e_type_conflict", "i_cur", p_wrong_type),
            prop("e_rep_as_cur", "i_cur", p_repealed, purpose="current_law"),
            prop("e_hist_unasked", "i_cur", p_repealed, purpose="repeal"),
            prop("e_hist_ok", "i_hist", p_repealed, purpose="repeal"),
            prop("e_comp_ok", "i_comp", p_wrong_type),
        ],
        issues=[
            IssueMapping(issue_id="i_cur", status="insufficient"),
            IssueMapping(issue_id="i_hist", status="supported", evidence_ids=["e_hist_ok"]),
            IssueMapping(issue_id="i_comp", status="supported", evidence_ids=["e_comp_ok"]),
        ],
    )
    coverage = validate_mapping(plan, output, catalog)
    rejected = {r.proposal_id: r.reason for r in coverage.rejections}
    assert rejected == {
        "e_meta": "evidence_document_metadata_mismatch",
        "e_role": "evidence_document_role_invalid",
        "e_notgt": "evidence_amendment_target_unknown",
        "e_wtgt": "evidence_document_scope_conflict",
        "e_future": "evidence_not_yet_effective",
        "e_type_conflict": "evidence_company_scope_conflict",
        "e_rep_as_cur": "evidence_not_current",
        "e_hist_unasked": "evidence_history_not_requested",
    }
    assert {e.evidence_id for e in coverage.evidence} == {"e_hist_ok", "e_comp_ok"}


def test_missing_information_requires_verified_source_condition_or_unresolved_scope():
    parent = make_parent(chunk_id="c1")
    supplied = make_supplied(parent, issue_ids=("i_cond", "i_scope", "i_bool"))
    q = "Chưa rõ loại hình doanh nghiệp và chưa biết Điều lệ công ty quy định thế nào về Điều 4?"
    f_cond = FactRecord(fact_id="f_cond", quote="chưa biết Điều lệ công ty quy định thế nào", start=34, end=76, is_unknown=True)
    f_scope = FactRecord(fact_id="f_scope", quote="Chưa rõ loại hình doanh nghiệp", start=0, end=30, is_unknown=True)
    amb_ref = ArticleReference(doc_id=None, dieu="4", quote="Điều 4", start=80, end=86, ambiguous=True, reason="document_unresolved")

    plan = make_plan(
        question=q,
        facts=[f_cond, f_scope],
        references=[amb_ref],
        issues=[
            IssueRecord(issue_id="i_cond", title="Điều kiện họp", fact_ids=["f_cond"], scope=IssueScope(company_types=["tnhh_2tv"]), main_query="họp"),
            IssueRecord(issue_id="i_scope", title="Phạm vi áp dụng", fact_ids=["f_scope"], scope=IssueScope(), main_query="áp dụng"),
            IssueRecord(issue_id="i_bool", title="Chỉ có boolean", fact_ids=["f_cond"], scope=IssueScope(company_types=["tnhh_2tv"]), main_query="họp"),
        ],
    )
    catalog = make_catalog([parent], supplied)
    u_ids = [u.unit_id for u in supplied[0].group.units]

    output = MapperOutput(
        evidence=[
            EvidenceProposal(evidence_id="e1", issue_id="i_cond", group_id="c1:g0", unit_ids=u_ids, doc_id="luat_dn_2020", dieu="58"),
            EvidenceProposal(evidence_id="e2", issue_id="i_bool", group_id="c1:g0", unit_ids=u_ids, doc_id="luat_dn_2020", dieu="58"),
        ],
        issues=[
            IssueMapping(
                issue_id="i_cond",
                status="conditional",
                evidence_ids=["e1"],
                missing_information=[
                    MissingProposal(
                        description="Cần cung cấp quy định tại Điều lệ công ty",
                        source_evidence_ids=["e1"],
                        condition_quote="Trừ trường hợp Điều lệ công ty có quy định khác",
                    )
                ],
            ),
            IssueMapping(
                issue_id="i_scope",
                status="ambiguous",
                missing_information=[
                    MissingProposal(
                        description="Cần xác định loại hình doanh nghiệp",
                        scope_field="company_types",
                        scope_fact_id="f_scope",
                    ),
                    MissingProposal(
                        description="Cần xác định văn bản của Điều 4",
                        scope_field="doc_ids",
                    ),
                ],
            ),
            IssueMapping(
                issue_id="i_bool",
                status="conditional",
                evidence_ids=["e2"],
                missing_information=[
                    MissingProposal(
                        description="Giả thuyết thiếu dữ kiện chỉ có cờ boolean",
                        source_grounded=True,
                        scope_unresolved=True,
                    )
                ],
            ),
        ],
    )
    coverage = validate_mapping(plan, output, catalog, id_prefix="m1:")
    by_id = {a.issue_id: a for a in coverage.assessments}
    assert by_id["i_cond"].status == IssueStatus.CONDITIONAL
    assert len(by_id["i_cond"].missing_information) == 1
    assert by_id["i_cond"].missing_information[0].source_grounded is True
    assert by_id["i_cond"].missing_information[0].source_evidence_ids == ["m1:e1"]

    assert by_id["i_scope"].status == IssueStatus.AMBIGUOUS
    assert len(by_id["i_scope"].missing_information) == 2
    assert all(m.scope_unresolved for m in by_id["i_scope"].missing_information)
    assert len(by_id["i_scope"].missing_scope) == 2

    assert by_id["i_bool"].status == IssueStatus.UNCERTAIN
    assert by_id["i_bool"].missing_information == []
    assert "missing_information_not_grounded" in by_id["i_bool"].reasons
    assert "conditional_information_not_grounded" in by_id["i_bool"].reasons


def test_no_certainty_promotion_and_conservative_whole_aggregation():
    parent = make_parent(chunk_id="c1")
    supplied = make_supplied(parent, issue_ids=("i1", "i2", "i3"))
    q = "Chưa rõ loại hình doanh nghiệp có được họp Hội đồng thành viên không?"
    f1 = FactRecord(fact_id="f1", quote=q, start=0, end=len(q), is_unknown=True)
    plan = make_plan(
        question=q,
        facts=[f1],
        issues=[
            IssueRecord(issue_id="i1", title="Vấn đề 1", fact_ids=["f1"], scope=IssueScope(), main_query="họp 1"),
            IssueRecord(issue_id="i2", title="Vấn đề 2", fact_ids=["f1"], scope=IssueScope(), main_query="họp 2"),
            IssueRecord(issue_id="i3", title="Vấn đề 3", fact_ids=["f1"], scope=IssueScope(), main_query="họp 3"),
        ],
        not_processed=[NotProcessedIssue(issue_id="i4", title="Vấn đề 4 vượt budget")],
    )
    catalog = make_catalog([parent], supplied)
    u_ids = [u.unit_id for u in supplied[0].group.units]

    output = MapperOutput(
        evidence=[
            EvidenceProposal(evidence_id="e1", issue_id="i1", group_id="c1:g0", unit_ids=u_ids, doc_id="luat_dn_2020", dieu="58"),
            EvidenceProposal(evidence_id="e2", issue_id="i2", group_id="c1:g0", unit_ids=u_ids, doc_id="luat_dn_2020", dieu="58"),
            EvidenceProposal(evidence_id="e3", issue_id="i3", group_id="c1:g0", unit_ids=u_ids, doc_id="luat_dn_2020", dieu="58"),
        ],
        issues=[
            IssueMapping(
                issue_id="i1", status="insufficient", evidence_ids=["e1"],
                missing_information=[MissingProposal(description="Điều lệ", source_evidence_ids=["e1"], condition_quote="Trừ trường hợp Điều lệ công ty có quy định khác")],
            ),
            IssueMapping(
                issue_id="i2", status="uncertain", evidence_ids=["e2"],
                missing_information=[MissingProposal(description="Loại hình", scope_field="company_types", scope_fact_id="f1")],
            ),
            IssueMapping(issue_id="i3", status="supported", evidence_ids=["e3"]),
        ],
    )
    coverage = validate_mapping(plan, output, catalog, unseen_issue_ids={"i3"})
    by_id = {a.issue_id: a for a in coverage.assessments}
    assert by_id["i1"].status == IssueStatus.INSUFFICIENT
    assert by_id["i2"].status == IssueStatus.UNCERTAIN
    assert by_id["i3"].status == IssueStatus.UNCERTAIN
    assert "source_spans_unseen" in by_id["i3"].reasons
    assert by_id["i4"].status == IssueStatus.NOT_PROCESSED

    whole, reasons = coverage_status(coverage)
    assert whole == ProcessingStatus.INSUFFICIENT
    assert set(reasons) == {PartialReason.EVIDENCE_MISSING, PartialReason.BUDGET_EXHAUSTED}

    all_sup = Coverage(coverage.evidence, (IssueAssessment(issue_id="i1", status=IssueStatus.SUPPORTED, evidence_ids=["e1"]),))
    assert coverage_status(all_sup) == (ProcessingStatus.SUPPORTED, [])

    not_assessed = Coverage((), (IssueAssessment(issue_id="i1", status=IssueStatus.NOT_ASSESSED),))
    assert coverage_status(not_assessed) == (ProcessingStatus.NOT_ASSESSED, [])


def test_merge_coverage_preserves_resolved_issues_and_accepted_evidence():
    parent = make_parent(chunk_id="c1")
    supplied = make_supplied(parent, issue_ids=("i_sup", "i_cond", "i_ins", "i_unc"))
    plan = make_plan(
        issues=[
            IssueRecord(issue_id="i_sup", title="Đã đủ", fact_ids=["f1"], main_query="q1"),
            IssueRecord(issue_id="i_cond", title="Có điều kiện", fact_ids=["f1"], main_query="q2"),
            IssueRecord(issue_id="i_ins", title="Thiếu căn cứ", fact_ids=["f1"], main_query="q3"),
            IssueRecord(issue_id="i_unc", title="Chưa chắc", fact_ids=["f1"], main_query="q4"),
        ],
        not_processed=[NotProcessedIssue(issue_id="i_np", title="Chưa xử lý")],
    )
    catalog = make_catalog([parent], supplied)
    u_ids = [u.unit_id for u in supplied[0].group.units]

    initial = validate_mapping(
        plan,
        MapperOutput(
            evidence=[
                EvidenceProposal(evidence_id="e_sup", issue_id="i_sup", group_id="c1:g0", unit_ids=u_ids, doc_id="luat_dn_2020", dieu="58"),
                EvidenceProposal(evidence_id="e_cond", issue_id="i_cond", group_id="c1:g0", unit_ids=u_ids, doc_id="luat_dn_2020", dieu="58"),
                EvidenceProposal(evidence_id="e_unc", issue_id="i_unc", group_id="c1:g0", unit_ids=u_ids, doc_id="luat_dn_2020", dieu="58"),
            ],
            issues=[
                IssueMapping(issue_id="i_sup", status="supported", evidence_ids=["e_sup"]),
                IssueMapping(
                    issue_id="i_cond", status="conditional", evidence_ids=["e_cond"],
                    missing_information=[MissingProposal(description="Điều lệ", source_evidence_ids=["e_cond"], condition_quote="Trừ trường hợp Điều lệ công ty có quy định khác")],
                ),
                IssueMapping(issue_id="i_ins", status="insufficient"),
                IssueMapping(issue_id="i_unc", status="supported", evidence_ids=["e_unc"]),
            ],
        ),
        catalog,
        unseen_issue_ids={"i_unc"},
        id_prefix="m1:",
    )
    rechecked = validate_mapping(
        plan,
        MapperOutput(
            evidence=[
                EvidenceProposal(evidence_id="e_ins2", issue_id="i_ins", group_id="c1:g0", unit_ids=u_ids, doc_id="luat_dn_2020", dieu="58"),
                EvidenceProposal(evidence_id="e_unc2", issue_id="i_unc", group_id="c1:g0", unit_ids=u_ids, doc_id="luat_dn_2020", dieu="58"),
            ],
            issues=[
                IssueMapping(issue_id="i_sup", status="insufficient"),
                IssueMapping(issue_id="i_cond", status="insufficient"),
                IssueMapping(issue_id="i_ins", status="supported", evidence_ids=["e_ins2"]),
                IssueMapping(issue_id="i_unc", status="supported", evidence_ids=["e_unc2"]),
            ],
        ),
        catalog,
        id_prefix="m2:",
    )
    merged = merge_coverage(initial, rechecked, {"i_sup", "i_cond", "i_ins", "i_unc", "i_np"})
    by_id = {a.issue_id: a for a in merged.assessments}
    assert by_id["i_sup"].status == IssueStatus.SUPPORTED
    assert by_id["i_sup"].evidence_ids == ["m1:e_sup"]
    assert by_id["i_cond"].status == IssueStatus.CONDITIONAL
    assert by_id["i_cond"].evidence_ids == ["m1:e_cond"]
    assert by_id["i_ins"].status == IssueStatus.SUPPORTED
    assert by_id["i_ins"].evidence_ids == ["m2:e_ins2"]
    assert by_id["i_unc"].status == IssueStatus.SUPPORTED
    assert by_id["i_unc"].evidence_ids == ["m1:e_unc", "m2:e_unc2"]
    assert by_id["i_np"].status == IssueStatus.NOT_PROCESSED
    assert {e.evidence_id for e in merged.evidence} == {"m1:e_sup", "m1:e_cond", "m1:e_unc", "m2:e_ins2", "m2:e_unc2"}


def test_direct_entrypoint_forwards_cli_arguments(monkeypatch):
    captured = []
    monkeypatch.setattr(sys, "argv", [__file__, "-q", "-p", "no:cacheprovider"])
    monkeypatch.setattr(pytest, "main", lambda args: captured.append(args) or 0)
    assert main() == 0
    assert captured == [[__file__, "-q", "-p", "no:cacheprovider"]]


def main() -> int:
    return int(pytest.main([__file__, *sys.argv[1:]]))


if __name__ == "__main__":
    raise SystemExit(main())
