"""Offline structured reference parsing and all-part lookup regressions."""
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest
from pydantic import ValidationError

from ragdn.article_references import ArticleReference, resolve_article_references
from ragdn.config import MANIFEST_PATH
from ragdn.manifest import load_manifest
from ragdn.retrieval import Retriever


LDN = "luat_dn_2020"
LAW76 = "luat_76_2025"


def pairs(question, manifest=None):
    return [(r.doc_id, r.dieu) for r in resolve_article_references(question, manifest)]


@pytest.mark.parametrize("question,expected", [
    ("Điều 58 và Điều 59 Luật Doanh nghiệp", [(LDN, "58"), (LDN, "59")]),
    ("Điều 58, 59 và 60 Luật Doanh nghiệp", [(LDN, "58"), (LDN, "59"), (LDN, "60")]),
    ("Luật Doanh nghiệp: Điều 58, Điều 59", [(LDN, "58"), (LDN, "59")]),
    ("Điều 3 Luật 76 và Điều 218 Luật Doanh nghiệp", [(LAW76, "3"), (LDN, "218")]),
    ("Luật 76: Điều 3 và Luật Doanh nghiệp: Điều 218", [(LAW76, "3"), (LDN, "218")]),
    ("Điều 1 Luật 76 và Điều 1 Luật Doanh nghiệp", [(LAW76, "1"), (LDN, "1")]),
    ("Điều 4 Nghị định 168 và Điều 5 Nghị định 47", [("nd_168_2025", "4"), ("nd_47_2021", "5")]),
    ("Điều 4 của Luật DN", [(LDN, "4")]),
    ("dieu 17a luat doanh nghiep", [(LDN, "17a")]),
    ("Điều 17d Luật Doanh nghiệp", [(LDN, "17d")]),
    ("Điều 17đ Luật Doanh nghiệp", [(LDN, "17đ")]),
    ("dieu 17dd Luat Doanh nghiep", [(LDN, "17đ")]),
    ("Điều 1 76/2025/QH15", [(LAW76, "1")]),
    ("Điều 58 Luật số 59/2020/QH14", [(LDN, "58")]),
    ("Điều 2 Nghị định số 168/2025/NĐ-CP", [("nd_168_2025", "2")]),
    ("Điều 128 Luật Doanh nghiệp (được sửa đổi bởi Luật 76)", [(LDN, "128")]),
    ("Điều 128 (được sửa đổi bởi Luật 76) Luật Doanh nghiệp", [(LDN, "128")]),
    ("Điều 1 Luật số 76/2025/QH15 sửa đổi, bổ sung một số điều của Luật Doanh nghiệp", [(LAW76, "1")]),
])
def test_local_document_article_binding(question, expected):
    records = resolve_article_references(question)
    assert [(r.doc_id, r.dieu) for r in records] == expected
    for reference in records:
        assert not reference.ambiguous and reference.reason is None
        assert reference.quote == question[reference.start:reference.end]


@pytest.mark.parametrize("question,expected", [
    ("Điều 4", [(None, "4")]),
    ("Điều 4 Luật không có trong manifest", [(None, "4")]),
    ("Điều 4 Luật 99 và Điều 58 Luật Doanh nghiệp", [(None, "4"), (LDN, "58")]),
    ("Điều 1 Luật Doanh nghiệp và Luật 76", [(None, "1")]),
    ("Luật Doanh nghiệp và Luật 76: Điều 1", [(None, "1")]),
    ("Luật 76 Điều 3 và Điều 218 Luật Doanh nghiệp", [(None, "3"), (None, "218")]),
    ("Điều 128 (Luật 76 sửa đổi)", [(None, "128")]),
    ("Điều 58 Luật Doanh nghiệp; Điều 1", [(LDN, "58"), (None, "1")]),
])
def test_uncertain_binding_is_visible_without_cross_product(question, expected):
    records = resolve_article_references(question)
    assert [(r.doc_id, r.dieu) for r in records] == expected
    for reference in records:
        if reference.doc_id is None:
            assert reference.ambiguous and reference.reason
        assert reference.quote == question[reference.start:reference.end]


@pytest.mark.parametrize("question", [
    "Luật 76 sửa đổi Luật Doanh nghiệp như thế nào?",
    "Tỷ lệ vốn 60% theo Luật Doanh nghiệp là bao nhiêu?",
    "Điều lệ doanh nghiệp quy định khác không?",
])
def test_document_mention_is_not_an_article_lookup(question):
    assert resolve_article_references(question) == []


def test_clause_point_and_original_unicode_provenance():
    question = "  Hãy tra điểm đ khoản 2 Điều 58 Luật Doanh nghiệp."
    reference, = resolve_article_references(question)
    assert (reference.diem, reference.khoan, reference.dieu) == ("đ", "2", "58")
    assert reference.quote == "điểm đ khoản 2 Điều 58 Luật Doanh nghiệp"
    assert reference.quote == question[reference.start:reference.end]
    decomposed = unicodedata.normalize("NFD", question)
    record, = resolve_article_references(decomposed)
    assert (record.diem, record.khoan) == ("đ", "2")
    assert record.quote == decomposed[record.start:record.end]
    assert unicodedata.normalize("NFC", record.quote) == reference.quote


def test_clause_before_article_unaccented():
    record, = resolve_article_references("diem a, khoan 2 dieu 59 Luat DN")
    assert (record.diem, record.khoan, record.dieu, record.doc_id) == ("a", "2", "59", LDN)


def test_local_qualifier_is_not_invented_for_later_group_members():
    records = resolve_article_references("Khoản 2 Điều 58 và Điều 59 Luật DN")
    assert [(r.dieu, r.khoan) for r in records] == [("58", "2"), ("59", None)]


def test_identical_manifest_aliases_are_ambiguous():
    documents = [{"doc_id": "a", "ten_ngan": "Luật Kiểm Thử"},
                 {"doc_id": "b", "ten_ngan": "Luật Kiểm Thử"}]
    record, = resolve_article_references("Điều 1 Luật Kiểm Thử", documents)
    assert record.doc_id is None and record.ambiguous
    assert record.reason == "conflicting_documents"


def test_percent_or_document_number_is_not_a_list_article():
    assert pairs("Điều 58 Luật DN và 60% vốn dự họp") == [(LDN, "58")]
    assert pairs("Điều 58 và 60% vốn dự họp Luật DN") == [(None, "58")]


@pytest.mark.parametrize("question", [
    "Điều 4 Luật Doanh nghiệp 2014",
    "Điều 4 Luật Doanh nghiệp năm 2014",
    "Điều 4 Luật Doanh nghiệp (2014)",
    "Điều 4 Luật 59/2014/QH13",
    "Điều 4 Luật Doanh nghiệp số 60/2020/QH14",
    "Điều 4 Luật Doanh nghiệp số 60",
])
def test_unsupported_document_edition_does_not_inherit_generic_alias(question):
    record, = resolve_article_references(question)
    assert record.doc_id is None and record.ambiguous and record.reason


def test_supported_explicit_year_is_retained_in_provenance():
    question = "Điều 4 Luật Doanh nghiệp năm 2020"
    record, = resolve_article_references(question)
    assert record.doc_id == LDN and record.quote == question


@pytest.mark.parametrize("question", [
    "Điều 1 Luật Doanh nghiệp và Luật 99",
    "Điều 1 Luật 99 và Luật Doanh nghiệp",
    "Điều 4 Luật Doanh nghiệp 2020 và Luật Doanh nghiệp 2014",
    "Điều 4 Luật Doanh nghiệp 2014 và Luật Doanh nghiệp 2020",
    "Điều 4 Luật Doanh nghiệp và Luật không có trong manifest",
    "Điều 4 59/2020/QH14 và 68/2014/QH13",
    "Điều 4 Luật Doanh nghiệp và 68/2014/QH13",
    "Điều 4 68/2014/QH13 và Luật Doanh nghiệp",
    "59/2020/QH14 và 68/2014/QH13: Điều 4",
])
def test_supported_and_unsupported_document_cluster_stays_unresolved(question):
    record, = resolve_article_references(question)
    assert record.doc_id is None and record.ambiguous
    assert record.quote == question


def test_manifest_inputs_and_call_time_loading(monkeypatch):
    custom = {"doc_id": "custom", "ten_ngan": "Luật Kiểm Thử", "so_hieu": "987/2099/QH99"}
    for manifest in (custom, [custom], {"documents": [custom]}):
        assert pairs("Điều 7 Luật Kiểm Thử", manifest) == [("custom", "7")]
        assert pairs("Điều 7 987/2099/QH99", manifest) == [("custom", "7")]
        assert pairs("Điều 7 Luật DN", manifest) == [(None, "7")]
    existing = load_manifest(MANIFEST_PATH)
    assert pairs("Điều 58 Luật DN", existing) == [(LDN, "58")]
    import ragdn.article_references as module
    monkeypatch.setattr(module, "load_manifest", lambda _: {"documents": [custom]})
    assert pairs("Điều 7 Luật Kiểm Thử") == [("custom", "7")]


def test_schema_forbids_unknown_fields_and_manifest_role_is_metadata():
    record, = resolve_article_references("Điều 1 Luật 76")
    assert record.role == "sua_doi" and record.amendment_target == LDN
    with pytest.raises(ValidationError):
        ArticleReference(**record.model_dump(), invented=True)


def make_lookup(chunks):
    retriever = object.__new__(Retriever)
    retriever.chunks = chunks
    retriever.by_article = {}
    for index, chunk in enumerate(chunks):
        retriever.by_article.setdefault((chunk["doc_id"], chunk["dieu"]), []).append(index)
    return retriever


def chunk(doc, article, part=0, **extra):
    return {"chunk_id": f"{doc}:d{article}:p{part}", "doc_id": doc, "dieu": article,
            "text": f"Synthetic source part {part}", **extra}


def test_lookup_all_nine_parts_stable_identity_no_mutation_and_no_duplicates():
    parents = [chunk(LDN, "58"), *[chunk(LAW76, "1", i) for i in range(9)], chunk(LDN, "1")]
    before = [dict(parent) for parent in parents]
    retriever = make_lookup(parents)
    selected = retriever.article_lookup_many("Điều 1 Luật 76 và Điều 58 Luật Doanh nghiệp và Điều 1 Luật 76")
    assert selected == parents[:-1]
    assert len(selected) == 10 and selected[-1]["chunk_id"].endswith(":p8")
    assert all(a is b for a, b in zip(selected, parents))
    assert parents == before
    assert retriever.article_lookup_many("Điều 1") == []


def test_lookup_suffix_exact_then_unaccented_fallback_and_repeal_eligibility():
    parents = [chunk(LDN, "17d"), chunk(LDN, "17đ", hieu_luc=False), chunk(LAW76, "17đ")]
    retriever = make_lookup(parents)
    assert retriever.article_lookup_many("dieu 17d Luat DN") == [parents[0]]
    assert retriever.article_lookup_many("dieu 17dd Luat DN") == [parents[1]]
    only_dd = make_lookup(parents[1:])
    assert only_dd.article_lookup_many("dieu 17d Luat DN") == [parents[1]]


def test_missing_document_never_defaults_even_with_one_matching_parent():
    retriever = make_lookup([chunk(LDN, "4")])
    assert retriever.article_lookup_many("Điều 4") == []


def main():
    return pytest.main([__file__, *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
