"""Offline P4-B retrieval orchestration, context packing and supplementary round tests."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import re
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pytest

from ragdn.evidence import EvidenceProposal, IssueMapping, MapperOutput, MissingProposal
from ragdn.query_config import QueryConfig
from ragdn.query_models import (
    FactProposal,
    IssueStatus,
    PartialReason,
    PlannerIssue,
    PlannerOutput,
    ProcessingMode,
    ProcessingStatus,
)
from ragdn.query_pipeline import (
    ContextView,
    MAPPER_SYSTEM,
    QueryResult,
    ReaderContext,
    build_mapper_prompt,
    run_query_pipeline,
)
from ragdn.retrieval import Retriever


class FakeTokenizer:
    def __init__(self, specials: int = 3):
        self.specials = specials

    def __call__(self, text, pair=None, *, add_special_tokens=False,
                 truncation=False, return_offsets_mapping=False, verbose=True):
        assert truncation is False
        spans = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        ids = list(range(len(spans)))
        if pair is not None:
            ids += list(range(len(re.findall(r"\S+", pair))))
        if add_special_tokens:
            ids += [999] * self.specials
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = spans
        return result


class FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.options = []
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.outputs:
            raise AssertionError("unexpected extra completion call")
        item = self.outputs.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, tuple):
            content, finish = item
        else:
            content, finish = item, "stop"
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content))]
        )


def make_chunk(
    index: int,
    text: str | None = None,
    *,
    doc_id: str = "luat_dn_2020",
    dieu: str | None = None,
    hieu_luc: bool = True,
    vai_tro: str = "chinh",
    loai_hinh: str | None = None,
    sua_doi_cho: str | None = None,
    ngay_hieu_luc: str | None = "2021-01-01",
    **extra,
) -> dict:
    chunk = {
        "chunk_id": f"c{index}",
        "doc_id": doc_id,
        "dieu": dieu or str(58 + index),
        "text": text if text is not None else (
            f"1. Quy định chi tiết tại nguồn số {index} áp dụng cho doanh nghiệp theo Luật này."
        ),
        "hieu_luc": hieu_luc,
        "vai_tro": vai_tro,
        "loai_hinh": loai_hinh,
        "sua_doi_cho": sua_doi_cho,
        "ngay_hieu_luc": ngay_hieu_luc,
    }
    chunk.update(extra)
    return chunk


def make_retriever(chunks: list[dict], *, raw_score: float = 0.8, score_fn=None) -> Retriever:
    instance = object.__new__(Retriever)
    instance.chunks = chunks
    instance.valid = np.array([bool(c.get("hieu_luc")) and c.get("vai_tro") in {"chinh", "sua_doi"} for c in chunks])
    instance.lh = np.array([c.get("loai_hinh") or "" for c in chunks])
    instance.by_article = {}
    for idx, chunk in enumerate(chunks):
        instance.by_article.setdefault((chunk["doc_id"], chunk["dieu"]), []).append(idx)
    base = np.arange(len(chunks), 0, -1, dtype=float)
    instance.emb = base.reshape(-1, 1)
    instance.embedder = SimpleNamespace(encode=Mock(return_value=np.array([1.0])))
    instance.bm25 = SimpleNamespace(get_scores=Mock(return_value=base))
    instance.bm25_noacc = SimpleNamespace(get_scores=Mock(return_value=base))
    predict = score_fn or (lambda pairs: np.full(len(pairs), raw_score, dtype=float))
    instance.reranker = SimpleNamespace(tokenizer=FakeTokenizer(), predict=Mock(side_effect=predict))
    instance.rerank_min = 0.15
    instance.glossary = {}
    instance.doc_aliases = [(r"luat doanh nghiep", "luat_dn_2020")]
    return instance


MANIFEST = [
    {"doc_id": "luat_dn_2020", "ten_vb": "Luật Doanh nghiệp", "ten_ngan": "Luật Doanh nghiệp",
     "so_hieu": "59/2020/QH14", "vai_tro": "chinh", "trang_thai": "con_hieu_luc", "ngay_hieu_luc": "2021-01-01"},
]


def planner_json(question: str, issues: list[dict], facts: list[dict] | None = None) -> str:
    payload = PlannerOutput(
        facts=[FactProposal(**f) for f in (facts or [{"fact_id": "f1", "quote": question}])],
        issues=[PlannerIssue(**item) for item in issues],
    )
    return payload.model_dump_json()


def test_reader_context_validation_unknown_and_oversize_prompt_fallback():
    with pytest.raises(ValueError, match="positive integer"):
        ReaderContext(context_tokens=0)
    with pytest.raises(ValueError, match="positive integer"):
        ReaderContext(context_tokens=True)
    with pytest.raises(ValueError, match="explicitly verified"):
        ReaderContext(context_tokens=4096, token_counter=len, tokenizer_verified=False)

    verified = ReaderContext(context_tokens=4096, token_counter=lambda s: len(s.split()), tokenizer_verified=True)
    assert verified.method == "verified_reader_tokenizer"
    byte_ctx = ReaderContext(context_tokens=8192)
    assert byte_ctx.method == "conservative_utf8_bytes"

    ret = make_retriever([make_chunk(0)])
    unknown_res = run_query_pipeline("Công ty cổ phần có quyền gì?", ret, manifest=MANIFEST)
    assert unknown_res.processing.mode == ProcessingMode.BASELINE_FALLBACK
    assert unknown_res.processing.route_reason == "context_budget_unknown"
    assert unknown_res.processing.status == ProcessingStatus.NOT_ASSESSED
    assert unknown_res.processing.warnings == ["context_budget_unknown"]

    tiny_ctx = ReaderContext(context_tokens=2600, token_counter=lambda s: 200, tokenizer_verified=True)
    oversize_res = run_query_pipeline("Công ty cổ phần có quyền gì?", ret, context=tiny_ctx, manifest=MANIFEST)
    assert oversize_res.processing.mode == ProcessingMode.BASELINE_FALLBACK
    assert oversize_res.processing.route_reason == "context_budget_exhausted"


def test_direct_whole_article_fastpath_and_truthful_oversize_or_unverified_lookup():
    c0 = make_chunk(0, dieu="58", text="1. Hội đồng thành viên gồm tất cả thành viên công ty là cá nhân.")
    c1 = make_chunk(1, dieu="58", text="2. Cuộc họp Hội đồng thành viên được triệu tập theo yêu cầu.")
    ret = make_retriever([c0, c1])
    ctx = ReaderContext(context_tokens=16384)
    q = "Điều 58 Luật Doanh nghiệp quy định gì?"

    direct = run_query_pipeline(q, ret, context=ctx, manifest=MANIFEST)
    assert direct.direct_complete is True
    assert direct.processing.mode == ProcessingMode.DIRECT
    assert direct.processing.status == ProcessingStatus.NOT_ASSESSED
    assert direct.processing.budget.completion_calls == 0
    assert direct.processing.budget.initial_queries == 1
    assert [s["chunk_id"] for s in direct.sources] == ["c0", "c1"]
    assert all(v.whole_parent for v in direct.context_views)

    tight_ctx = ReaderContext(
        context_tokens=2600,
        token_counter=lambda s: 100 if "Trích nguyên văn" in s else 10,
        tokenizer_verified=True,
    )
    client = FakeClient([
        planner_json(q, [{"issue_id": "i1", "title": "Điều 58", "fact_ids": ["f1"],
                          "main_query": "Điều 58 Luật Doanh nghiệp", "backup_query": "quy định Điều 58"}]),
        MapperOutput(
            evidence=[
                EvidenceProposal(evidence_id="e1", issue_id="i1", group_id="c0:g0", unit_ids=["c0:u0"], doc_id="luat_dn_2020", dieu="58"),
            ],
            issues=[IssueMapping(issue_id="i1", status="supported", evidence_ids=["e1"])],
        ).model_dump_json(),
    ])
    enhanced = run_query_pipeline(q, ret, context=tight_ctx, manifest=MANIFEST, client=client)
    assert enhanced.direct_complete is False
    assert enhanced.processing.mode == ProcessingMode.DIRECT
    assert enhanced.processing.route_reason == "context_budget_exhausted"
    assert enhanced.processing.budget.completion_calls == 0
    assert client.calls == []

    repealed_ret = make_retriever([make_chunk(0, dieu="58", hieu_luc=False)])
    client_rep = FakeClient([
        planner_json(q, [{"issue_id": "i1", "title": "Điều 58", "fact_ids": ["f1"],
                          "main_query": "Điều 58 Luật Doanh nghiệp", "backup_query": "quy định Điều 58"}]),
        MapperOutput(issues=[IssueMapping(issue_id="i1", status="insufficient")]).model_dump_json(),
    ])
    rep_res = run_query_pipeline(q, repealed_ret, context=ctx, manifest=MANIFEST, client=client_rep)
    assert rep_res.direct_complete is False
    assert rep_res.processing.mode == ProcessingMode.DIRECT
    assert "lookup_source_not_current" in rep_res.processing.warnings
    assert client_rep.calls == []


def test_initial_and_supplementary_accounting_and_accepted_issue_preservation():
    c0 = make_chunk(0, dieu="58", loai_hinh="tnhh_2tv", text="1. Hội đồng thành viên họp ít nhất mỗi năm một lần.")
    c1 = make_chunk(1, dieu="115", loai_hinh="ctcp", text="1. Cổ đông phổ thông có quyền tham dự và phát biểu tại Đại hội đồng cổ đông.")
    ret = make_retriever([c0, c1])
    ctx = ReaderContext(context_tokens=32768)
    q = "Công ty TNHH hai thành viên họp thế nào và Công ty cổ phần có quyền gì?"
    f1_quote = "Công ty TNHH hai thành viên họp thế nào"
    f2_quote = "Công ty cổ phần có quyền gì"

    plan_str = planner_json(
        q,
        facts=[
            {"fact_id": "f1", "quote": f1_quote},
            {"fact_id": "f2", "quote": f2_quote},
        ],
        issues=[
            {"issue_id": "i1", "title": "Họp TNHH", "fact_ids": ["f1"],
             "scope": {"company_types": ["tnhh_2tv"]}, "main_query": "họp Hội đồng thành viên TNHH hai thành viên",
             "backup_query": "triệu tập họp TNHH hai thành viên"},
            {"issue_id": "i2", "title": "Quyền CTCP", "fact_ids": ["f2"],
             "scope": {"company_types": ["ctcp"]}, "main_query": "quyền cổ đông công ty cổ phần",
             "backup_query": "biểu quyết đại hội đồng cổ đông công ty cổ phần"},
        ],
    )
    m1_str = MapperOutput(
        evidence=[
            EvidenceProposal(evidence_id="e1", issue_id="i1", group_id="c0:g0", unit_ids=["c0:u0"], doc_id="luat_dn_2020", dieu="58", khoan="1"),
        ],
        issues=[
            IssueMapping(issue_id="i1", status="supported", evidence_ids=["e1"]),
            IssueMapping(issue_id="i2", status="insufficient"),
        ],
    ).model_dump_json()

    c2 = make_chunk(2, dieu="116", loai_hinh="ctcp", text="1. Cổ đông ưu đãi biểu quyết có số phiếu biểu quyết nhiều hơn.")
    ret.chunks.append(c2)
    ret.valid = np.array([True, True, True])
    ret.lh = np.array(["", "", "ctcp"])
    ret.by_article[("luat_dn_2020", "116")] = [2]
    ret.emb = np.array([[3.0], [2.0], [1.0]])
    ret.bm25 = SimpleNamespace(get_scores=Mock(side_effect=[
        np.array([3.0, 2.0, -np.inf]),
        np.array([3.0, 2.0, -np.inf]),
        np.array([3.0, 2.0, -np.inf]),
        np.array([-np.inf, 1.0, 5.0]),
    ]))
    ret.bm25_noacc = SimpleNamespace(get_scores=Mock(side_effect=[
        np.array([3.0, 2.0, -np.inf]),
        np.array([3.0, 2.0, -np.inf]),
        np.array([3.0, 2.0, -np.inf]),
        np.array([-np.inf, 1.0, 5.0]),
    ]))

    m2_str = MapperOutput(
        evidence=[
            EvidenceProposal(evidence_id="e2", issue_id="i2", group_id="c2:g0", unit_ids=["c2:u0"], doc_id="luat_dn_2020", dieu="116", khoan="1"),
        ],
        issues=[
            IssueMapping(issue_id="i1", status="insufficient"),
            IssueMapping(issue_id="i2", status="supported", evidence_ids=["e2"]),
        ],
    ).model_dump_json()

    client = FakeClient([plan_str, m1_str, m2_str])
    res = run_query_pipeline(q, ret, config=QueryConfig(parents_per_query=2, context_tokens=32768), context=ctx, manifest=MANIFEST, client=client)
    assert res.processing.status == ProcessingStatus.SUPPORTED
    assert res.processing.budget.completion_calls == 3
    assert res.processing.budget.initial_queries == 3
    assert res.processing.budget.supplementary_rounds == 1
    assert res.processing.budget.supplementary_queries == 1
    by_id = {a.issue_id: a for a in res.processing.issues}
    assert by_id["i1"].status == IssueStatus.SUPPORTED
    assert by_id["i1"].evidence_ids == ["m1:e1"]
    assert by_id["i2"].status == IssueStatus.SUPPORTED
    assert by_id["i2"].evidence_ids == ["m2:e2"]


def test_supplementary_skipped_for_conditional_or_ambiguous_only_and_no_recheck_without_new_spans():
    c0 = make_chunk(
        0, dieu="58",
        text="1. Trừ trường hợp Điều lệ công ty có quy định khác, cuộc họp được tiến hành khi có số thành viên dự họp.",
    )
    ret = make_retriever([c0])
    ctx = ReaderContext(context_tokens=32768)
    q = "Chưa rõ loại hình doanh nghiệp và chưa biết Điều lệ công ty quy định thế nào về cuộc họp?"
    f1 = "Chưa rõ loại hình doanh nghiệp"
    f2 = "chưa biết Điều lệ công ty quy định thế nào"

    plan_str = planner_json(
        q,
        facts=[
            {"fact_id": "f1", "quote": f1, "is_unknown": True},
            {"fact_id": "f2", "quote": f2, "is_unknown": True},
        ],
        issues=[
            {"issue_id": "i1", "title": "Điều kiện họp", "fact_ids": ["f2"],
             "main_query": "điều kiện tiến hành cuộc họp", "backup_query": "triệu tập cuộc họp"},
            {"issue_id": "i2", "title": "Phạm vi loại hình", "fact_ids": ["f1"],
             "main_query": "loại hình doanh nghiệp họp", "backup_query": "quy định loại hình doanh nghiệp"},
        ],
    )
    m1_cond_amb = MapperOutput(
        evidence=[
            EvidenceProposal(evidence_id="e1", issue_id="i1", group_id="c0:g0", unit_ids=["c0:u0"], doc_id="luat_dn_2020", dieu="58", khoan="1"),
        ],
        issues=[
            IssueMapping(
                issue_id="i1", status="conditional", evidence_ids=["e1"],
                missing_information=[MissingProposal(description="Cần Điều lệ", source_evidence_ids=["e1"], condition_quote="Trừ trường hợp Điều lệ công ty có quy định khác")],
            ),
            IssueMapping(
                issue_id="i2", status="ambiguous",
                missing_information=[MissingProposal(description="Cần loại hình", scope_field="company_types", scope_fact_id="f1")],
            ),
        ],
    ).model_dump_json()

    client = FakeClient([plan_str, m1_cond_amb])
    res = run_query_pipeline(q, ret, context=ctx, manifest=MANIFEST, client=client)
    assert res.processing.budget.completion_calls == 2
    assert res.processing.budget.supplementary_rounds == 0
    assert res.processing.status == ProcessingStatus.PARTIAL
    assert set(res.partial_reasons) == {PartialReason.INFORMATION_MISSING, PartialReason.AMBIGUOUS_SCOPE}

    m1_ins = MapperOutput(
        evidence=[
            EvidenceProposal(evidence_id="e1", issue_id="i1", group_id="c0:g0", unit_ids=["c0:u0"], doc_id="luat_dn_2020", dieu="58", khoan="1"),
        ],
        issues=[
            IssueMapping(issue_id="i1", status="supported", evidence_ids=["e1"]),
            IssueMapping(issue_id="i2", status="insufficient"),
        ],
    ).model_dump_json()
    client_no_new = FakeClient([plan_str, m1_ins])
    res_no_new = run_query_pipeline(q, ret, context=ctx, manifest=MANIFEST, client=client_no_new)
    assert res_no_new.processing.budget.supplementary_rounds == 1
    assert res_no_new.processing.budget.completion_calls == 2
    assert "supplementary_no_new_source_spans" in res_no_new.processing.warnings


def test_same_parent_new_offset_triggers_recheck_and_reopens_unseen_windows():
    long_text = (
        "1. " + " ".join(f"tu_khoan_mot_{i}" for i in range(21)) + ".\n"
        "2. " + " ".join(f"tu_khoan_hai_{i}" for i in range(21)) + "."
    )
    c0 = make_chunk(0, dieu="58", text=long_text)
    ret = make_retriever([c0])
    ctx = ReaderContext(context_tokens=32768)
    q = "Quy định về biểu quyết của công ty là gì?"
    cfg = QueryConfig(
        passage_tokens=22, overlap_tokens=0, query_tokens=16, pair_tokens=64,
        initial_rerank_pairs=1, supplementary_rerank_pairs=3, max_rerank_pairs=4,
        context_tokens=32768,
    )
    plan_str = planner_json(
        q,
        issues=[{"issue_id": "i1", "title": "Biểu quyết", "fact_ids": ["f1"],
                 "main_query": "quy định biểu quyết công ty", "backup_query": "quy định biểu quyết công ty"}],
    )
    m1_str = MapperOutput(
        evidence=[
            EvidenceProposal(evidence_id="e1", issue_id="i1", group_id="c0:g0", unit_ids=["c0:u0"], doc_id="luat_dn_2020", dieu="58", khoan="1"),
        ],
        issues=[IssueMapping(issue_id="i1", status="supported", evidence_ids=["e1"])],
    ).model_dump_json()
    m2_str = MapperOutput(
        evidence=[
            EvidenceProposal(evidence_id="e2", issue_id="i1", group_id="c0:g1", unit_ids=["c0:u1"], doc_id="luat_dn_2020", dieu="58", khoan="2"),
        ],
        issues=[IssueMapping(issue_id="i1", status="supported", evidence_ids=["e2"])],
    ).model_dump_json()

    client = FakeClient([plan_str, m1_str, m2_str])
    res = run_query_pipeline(q, ret, config=cfg, context=ctx, manifest=MANIFEST, client=client)
    assert res.processing.budget.initial_rerank_pairs == 1
    assert res.processing.budget.supplementary_rerank_pairs >= 1
    assert res.processing.budget.supplementary_rounds == 1
    assert res.processing.budget.completion_calls == 3
    assert res.processing.status == ProcessingStatus.SUPPORTED
    assert res.processing.issues[0].evidence_ids == ["m1:e1", "m2:e2"]
    assert len(res.context_views) == 1
    assert res.context_views[0].whole_parent is True


def test_mapper_or_recheck_failure_discards_coverage_and_reuses_legacy_once():
    c0 = make_chunk(0, dieu="58", text="1. Hội đồng thành viên họp ít nhất mỗi năm một lần.")
    c1 = make_chunk(1, dieu="59", text="1. Triệu tập họp Hội đồng thành viên theo yêu cầu.")
    ret = make_retriever([c0, c1])
    ret.retrieve = Mock(return_value={"hits": [0], "max_score": 0.7, "low_confidence": False, "flag": None, "loai_hinh": None})
    ctx = ReaderContext(context_tokens=32768)
    q = "Hội đồng thành viên họp và triệu tập thế nào?"

    plan_str = planner_json(
        q,
        issues=[{"issue_id": "i1", "title": "Họp", "fact_ids": ["f1"],
                 "main_query": "Hội đồng thành viên họp", "backup_query": "triệu tập họp Hội đồng thành viên"}],
    )
    m1_ins = MapperOutput(
        evidence=[
            EvidenceProposal(evidence_id="e1", issue_id="i1", group_id="c0:g0", unit_ids=["c0:u0"], doc_id="luat_dn_2020", dieu="58", khoan="1"),
        ],
        issues=[IssueMapping(issue_id="i1", status="insufficient", evidence_ids=["e1"])],
    ).model_dump_json()

    ret.bm25 = SimpleNamespace(get_scores=Mock(side_effect=[np.array([2.0, 0.0]), np.array([2.0, 0.0]), np.array([0.0, 5.0])]))
    ret.bm25_noacc = SimpleNamespace(get_scores=Mock(side_effect=[np.array([2.0, 0.0]), np.array([2.0, 0.0]), np.array([0.0, 5.0])]))
    ret.emb = np.array([[2.0], [0.0]])

    client_recheck_fail = FakeClient([plan_str, m1_ins, httpx.ConnectError("connection dropped")])
    precomputed_legacy = {"hits": [1], "max_score": 0.9, "low_confidence": False, "flag": "cached", "loai_hinh": None}
    res = run_query_pipeline(
        q, ret, config=QueryConfig(parents_per_query=1, context_tokens=32768),
        context=ctx, manifest=MANIFEST, client=client_recheck_fail, legacy_result=precomputed_legacy,
    )
    assert res.processing.mode == ProcessingMode.BASELINE_FALLBACK
    assert res.processing.status == ProcessingStatus.NOT_ASSESSED
    assert res.processing.route_reason == "structured_service_error"
    assert res.processing.warnings == ["structured_service_error"]
    assert res.evidence == ()
    assert res.plan is None
    assert res.processing.budget.completion_calls == 3
    assert all(opt["max_retries"] == 0 for opt in client_recheck_fail.options)
    ret.retrieve.assert_not_called()

    ticks = iter([0.0, 0.0, 601.0, 601.0])
    deadline_res = run_query_pipeline(q, ret, context=ctx, manifest=MANIFEST, clock=lambda: next(ticks, 601.0))
    assert deadline_res.processing.mode == ProcessingMode.BASELINE_FALLBACK
    assert deadline_res.processing.route_reason == "stage_deadline_exhausted"
    ret.retrieve.assert_not_called()

    broken_ctx = ReaderContext(context_tokens=32768, token_counter=lambda s: 1 / 0, tokenizer_verified=True)
    with pytest.raises(ZeroDivisionError):
        run_query_pipeline(q, ret, context=broken_ctx, manifest=MANIFEST)


def test_budget_overflow_visibility_and_repeated_clause_atomic_group_packing():
    rep_text = (
        "1. Hội đồng thành viên quyết định chiến lược phát triển của công ty.\n"
        "1. Hội đồng thành viên quyết định kế hoạch kinh doanh hằng năm của công ty."
    )
    chunks = [
        make_chunk(0, dieu="58", text=rep_text),
        make_chunk(1, dieu="59", text="1. Triệu tập họp Hội đồng thành viên theo quy định."),
        make_chunk(2, dieu="60", text="1. Điều kiện và thể thức tiến hành họp Hội đồng thành viên."),
    ]
    ret = make_retriever(chunks)
    ret.bm25 = SimpleNamespace(get_scores=Mock(side_effect=[np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0])]))
    ret.bm25_noacc = SimpleNamespace(get_scores=Mock(side_effect=[np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0])]))
    ctx = ReaderContext(context_tokens=65536)
    q = "Các vấn đề quản trị công ty theo quy định là gì?"

    seven_issues = [
        {"issue_id": f"i{k}", "title": f"Vấn đề quản trị công ty", "fact_ids": ["f1"],
         "main_query": f"vấn đề quản trị công ty", "backup_query": f"quy định quản trị công ty"}
        for k in range(1, 8)
    ]
    plan_str = planner_json(q, issues=seven_issues)
    m1_str = MapperOutput(
        evidence=[
            EvidenceProposal(evidence_id="e1", issue_id="i1", group_id="c0:g0", unit_ids=["c0:u0", "c0:u1"], doc_id="luat_dn_2020", dieu="58"),
        ],
        issues=[IssueMapping(issue_id=f"i{k}", status="supported" if k == 1 else "insufficient", evidence_ids=["e1"] if k == 1 else []) for k in range(1, 7)],
    ).model_dump_json()

    cfg = QueryConfig(
        parents_per_query=2, max_active_parents=2, max_evidence_groups=1, max_supplementary_rounds=0, context_tokens=65536,
    )
    client = FakeClient([plan_str, m1_str])
    res = run_query_pipeline(q, ret, config=cfg, context=ctx, manifest=MANIFEST, client=client)
    assert "issue_budget_exhausted" in res.processing.warnings
    assert "active_parent_budget_exhausted" in res.processing.warnings
    assert "evidence_group_budget_exhausted" in res.processing.warnings
    assert res.processing.budget.not_processed == ["i7"]
    assert res.diagnostics["groups"] == ("c0:g0",)
    assert res.context_views[0].whole_parent is True
    assert res.processing.status == ProcessingStatus.INSUFFICIENT
    assert PartialReason.BUDGET_EXHAUSTED in res.partial_reasons


def test_bcd_internal_variants_share_budgets_and_report_not_assessed():
    chunks = [make_chunk(0, dieu="58"), make_chunk(1, dieu="59")]
    ret = make_retriever(chunks)
    ctx = ReaderContext(context_tokens=32768)
    q = "Quy định về họp và biểu quyết của công ty là gì?"

    res_b = run_query_pipeline(q, ret, context=ctx, manifest=MANIFEST, variant="B")
    assert res_b.processing.mode == ProcessingMode.ENHANCED
    assert res_b.processing.status == ProcessingStatus.NOT_ASSESSED
    assert res_b.processing.budget.completion_calls == 0
    assert res_b.processing.budget.initial_queries == 1

    plan_str = planner_json(
        q,
        issues=[
            {"issue_id": "i1", "title": "Họp", "fact_ids": ["f1"], "main_query": "quy định về họp công ty", "backup_query": "triệu tập họp"},
            {"issue_id": "i2", "title": "Biểu quyết", "fact_ids": ["f1"], "main_query": "biểu quyết của công ty", "backup_query": "thông qua nghị quyết"},
        ],
    )
    client_c = FakeClient([plan_str])
    res_c = run_query_pipeline(q, ret, context=ctx, manifest=MANIFEST, client=client_c, variant="C")
    assert res_c.processing.status == ProcessingStatus.NOT_ASSESSED
    assert res_c.processing.budget.completion_calls == 1
    assert res_c.processing.budget.initial_queries == 3

    m1_str = MapperOutput(
        evidence=[
            EvidenceProposal(evidence_id="e1", issue_id="i1", group_id="c0:g0", unit_ids=["c0:u0"], doc_id="luat_dn_2020", dieu="58", khoan="1"),
            EvidenceProposal(evidence_id="e2", issue_id="i2", group_id="c1:g0", unit_ids=["c1:u0"], doc_id="luat_dn_2020", dieu="59", khoan="1"),
        ],
        issues=[
            IssueMapping(issue_id="i1", status="supported", evidence_ids=["e1"]),
            IssueMapping(issue_id="i2", status="supported", evidence_ids=["e2"]),
        ],
    ).model_dump_json()
    client_d = FakeClient([m1_str])
    res_d = run_query_pipeline(q, ret, context=ctx, manifest=MANIFEST, client=client_d, variant="D", cached_plan=res_c.plan)
    assert res_d.processing.status == ProcessingStatus.SUPPORTED
    assert res_d.processing.budget.completion_calls == 1


def test_concurrent_requests_isolation_source_immutability_and_frozen_citation_order():
    chunks = [make_chunk(0, dieu="58"), make_chunk(1, dieu="59")]
    snapshot = deepcopy(chunks)
    ret = make_retriever(chunks)
    ctx = ReaderContext(context_tokens=32768)

    def run_one(question: str, chunk_idx: int, dieu: str):
        cid = f"c{chunk_idx}"
        client = FakeClient([
            planner_json(question, [{"issue_id": "i1", "title": "Tra cứu", "fact_ids": ["f1"],
                                     "main_query": question, "backup_query": question}]),
            MapperOutput(
                evidence=[
                    EvidenceProposal(evidence_id="e1", issue_id="i1", group_id=f"{cid}:g0", unit_ids=[f"{cid}:u0"], doc_id="luat_dn_2020", dieu=dieu, khoan="1"),
                ],
                issues=[IssueMapping(issue_id="i1", status="supported", evidence_ids=["e1"])],
            ).model_dump_json(),
        ])
        return run_query_pipeline(question, ret, context=ctx, manifest=MANIFEST, client=client)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(
            lambda args: run_one(*args),
            [("Công ty A họp thế nào?", 0, "58"), ("Công ty B biểu quyết thế nào?", 1, "59")],
        ))

    assert first.processing.trace_id != second.processing.trace_id
    assert first.original_question != second.original_question
    assert first.evidence[0].parent_chunk_id == "c0"
    assert second.evidence[0].parent_chunk_id == "c1"
    assert ret.chunks == snapshot
    for idx, (src, view) in enumerate(zip(first.sources, first.context_views), 1):
        assert view.citation_index == idx
        assert view.parent_chunk_id == src["chunk_id"]
        assert first.diagnostics["seen_spans"][src["chunk_id"]] == tuple((s.start, s.end) for s in view.spans)
    with pytest.raises(TypeError):
        first.sources[0]["text"] = "mutated"


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
