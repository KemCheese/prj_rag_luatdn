"""Offline candidate, RAM-window and request-local rerank contracts."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from copy import deepcopy
import re
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from ragdn.article_references import ArticleReference
from ragdn.retrieval import Retriever, pool_candidates
from ragdn.source_windows import (
    UnsupportedTokenizerError, build_source_windows, encoded, pair_token_count,
    prepare_query, source_groups,
)


class Tokenizer:
    def __init__(self, specials=3):
        self.specials = specials

    def __call__(self, text, pair=None, *, add_special_tokens=False,
                 truncation=False, return_offsets_mapping=False):
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


def chunk(index, text=None, **overrides):
    result = dict(chunk_id=f"c{index}", doc_id="luat_dn_2020", dieu=str(index + 1),
                  text=text or f"Nội dung nguồn {index}.", hieu_luc=True,
                  vai_tro="chinh", loai_hinh=None)
    result.update(overrides)
    return result


def retriever(chunks, scores=None, raw=0.5):
    instance = object.__new__(Retriever)
    instance.chunks = chunks
    instance.valid = np.array([c["hieu_luc"] and c["vai_tro"] in {"chinh", "sua_doi"} for c in chunks])
    instance.lh = np.array([c.get("loai_hinh") or "" for c in chunks])
    instance.by_article = {}
    for index, parent in enumerate(chunks):
        instance.by_article.setdefault((parent["doc_id"], parent["dieu"]), []).append(index)
    values = np.asarray(scores if scores is not None else np.arange(len(chunks), 0, -1), dtype=float)
    instance.emb = values.reshape(-1, 1)
    instance.embedder = SimpleNamespace(encode=Mock(return_value=np.array([1.])))
    instance.bm25 = SimpleNamespace(get_scores=Mock(return_value=values))
    instance.bm25_noacc = SimpleNamespace(get_scores=Mock(return_value=values))
    instance.reranker = SimpleNamespace(tokenizer=Tokenizer(),
        predict=Mock(side_effect=lambda pairs: np.full(len(pairs), raw)))
    instance.rerank_min = 0.15
    instance.glossary = {}
    instance.doc_aliases = []
    return instance


def batch(instance, query="quyền thành viên", query_id="q1", issue_id="i1", **kwargs):
    kwargs.setdefault("original_question", query)
    kwargs.setdefault("references", [])
    return instance.candidate_batch(query, query_id=query_id, issue_id=issue_id, **kwargs)


def reference(question, **kwargs):
    fields = dict(doc_id="luat_dn_2020", dieu="1", khoan=None, diem=None,
                  quote=question, start=0, end=len(question), ambiguous=False, reason=None)
    fields.update(kwargs)
    return ArticleReference(**fields)


def test_three_shared_channels_include_unlabelled_and_amendments():
    parents = [chunk(0), chunk(1, vai_tro="sua_doi"), chunk(2, loai_hinh="ctcp"),
               chunk(3, hieu_luc=False), chunk(4, vai_tro="tham_khao")]
    instance = retriever(parents)
    result = batch(instance)
    assert [c.chunk_id for c in result.candidates] == ["c0", "c1", "c2"]
    assert set(result.channel_available) == {"dense", "bm25", "bm25_noacc"}
    assert result.channel_available == dict.fromkeys(result.channel_available, 3)
    assert all(c.channels == ("dense", "bm25", "bm25_noacc") for c in result.candidates)
    assert result.candidates[0].rrf_score == pytest.approx(3 / 61)
    instance.bm25.get_scores.assert_called_once()
    instance.bm25_noacc.get_scores.assert_called_once()


@pytest.mark.parametrize("company_type,quote,allowed", [
    ("ctcp", "Công ty cổ phần", {None, "ctcp"}),
    ("dnnn", "Doanh nghiệp nhà nước", {None, "dnnn", "ctcp", "tnhh_mtv", "tnhh_2tv"}),
    ("nhom_cty", "Nhóm công ty", {None, "dnnn", "ctcp", "tnhh_mtv", "tnhh_2tv", "dntn"}),
])
def test_explicit_type_branch_keeps_shared_channels_and_compatible_types(company_type, quote, allowed):
    labels = [None, "ctcp", "dnnn", "tnhh_mtv", "tnhh_2tv", "dntn"]
    instance = retriever([chunk(index, loai_hinh=label) for index, label in enumerate(labels)])
    result = batch(instance, query="quyền", original_question=f"{quote} có quyền gì?",
                   company_type=company_type, company_type_quote=quote)
    assert len(result.candidates) == len(labels)
    for candidate in result.candidates:
        assert ("dense_type" in candidate.channels) == (labels[candidate.chunk_index] in allowed)
        assert len(set(candidate.channels) & {"dense", "bm25", "bm25_noacc"}) == 3


@pytest.mark.parametrize("quote", [None, "HDQT", "Công ty cổ phần"])
def test_type_branch_requires_literal_explicit_original_scope(quote):
    instance = retriever([chunk(0)])
    with pytest.raises(ValueError, match="type scope"):
        batch(instance, query="HDQT", original_question="HDQT có quyền gì?",
              company_type="ctcp", company_type_quote=quote)
    instance.embedder.encode.assert_not_called()


def test_whole_question_type_is_not_silently_imposed_on_issue():
    instance = retriever([chunk(0, loai_hinh="dntn"), chunk(1, loai_hinh="ctcp")])
    result = batch(instance, query="thủ tục", original_question="Công ty cổ phần có thủ tục gì?")
    assert set(result.channel_available) == {"dense", "bm25", "bm25_noacc"}


def test_candidate_limits_rrf_provenance_and_visible_unseen():
    instance = retriever([chunk(index) for index in range(30)])
    result = batch(instance, query_id="q-original", issue_id="i-a")
    assert len(result.candidates) == 12
    assert len(result.unseen_parent_ids) == 8
    assert result.channel_unseen == {"dense": 10, "bm25": 10, "bm25_noacc": 10}
    assert all(c.query_id == "q-original" and c.issue_id == "i-a" for c in result.candidates)
    assert result.candidates[-1].rrf_rank == 12
    assert result.candidates[-1].rrf_score == pytest.approx(3 / 72)
    lower = batch(instance, per_channel=4, per_query=2)
    assert len(lower.candidates) == 2 and len(lower.unseen_parent_ids) == 2


def test_direct_all_parts_precede_pool_budget_and_repealed_sources_remain():
    question = "khoản 2 Điều 1 Luật Doanh nghiệp"
    parents = [chunk(index, text="1. Điều kiện chung.\n2. Điều kiện riêng.", dieu="1", hieu_luc=False)
               for index in range(16)]
    instance = retriever(parents)
    before = deepcopy(parents)
    result = batch(instance, question, references=[reference(question, khoan="2")])
    assert len(result.candidates) == 16
    assert all(c.direct and c.targeted and c.rrf_score is None for c in result.candidates)
    assert all(instance.chunks[c.chunk_index] is parents[c.chunk_index] for c in result.candidates)
    pool = pool_candidates([result], max_parents=5)
    assert len(pool.parents) == 5 and len(pool.excluded_parent_ids) == 11
    assert pool.unseen_parent_ids == pool.excluded_parent_ids
    assert parents == before


def test_targeted_direct_parts_receive_priority():
    question = "khoản 2 Điều 1"
    instance = retriever([chunk(0, text="1. Phần đầu.", dieu="1"),
                          chunk(1, text="2. Phần đích.", dieu="1")])
    result = batch(instance, question, references=[reference(question, khoan="2")])
    assert [p.chunk_id for p in pool_candidates([result], max_parents=1).parents] == ["c1"]


def test_unresolved_and_unavailable_references_are_explicit():
    question = "Điều 99"
    instance = retriever([chunk(0)])
    result = batch(instance, question, references=[reference(question, doc_id=None, ambiguous=True,
                        reason="document_unresolved"), reference(question, dieu="99")])
    assert set(result.warnings) == {"reference_unresolved", "reference_parent_unavailable"}
    assert len(result.references) == 2


def test_pool_fairness_dedup_and_scores_not_added_across_queries():
    instance = retriever([chunk(index) for index in range(40)])
    a = batch(instance, query_id="qa", issue_id="ia")
    b = batch(instance, query_id="qb", issue_id="ia")
    instance.emb = np.arange(40).reshape(-1, 1)
    instance.bm25.get_scores.return_value = np.arange(40)
    instance.bm25_noacc.get_scores.return_value = np.arange(40)
    c = batch(instance, query_id="qc", issue_id="ib")
    result = pool_candidates([a, b, c], max_parents=4)
    assert [p.chunk_id for p in result.parents] == ["c0", "c39", "c1", "c38"]
    assert len(result.parents[0].nominations) == 2
    assert result.parents[0].nominations[0].rrf_score == pytest.approx(3 / 61)
    assert result.parents[0].nominations[1].rrf_score == pytest.approx(3 / 61)
    assert len(set(p.chunk_id for p in result.parents)) == 4


def test_long_parent_end_exact_offsets_and_overlapping_windows():
    parent = chunk(0, text="  " + " ".join(f"t{index}" for index in range(1800)) + "  cuối")
    tokenizer = Tokenizer()
    result = build_source_windows(parent, tokenizer)
    assert len(result) >= 6 and result[-1].end == len(parent["text"])
    assert result[-1].text.endswith("cuối")
    for view in result:
        assert view.text == parent["text"][view.start:view.end]
        assert view.token_count <= 384
    assert all(left.end > right.start for left, right in zip(result, result[1:]))


def test_query_budget_uses_offsets_without_mutating_original():
    tokenizer = Tokenizer()
    query = " ".join(f"q{index}" for index in range(140))
    shortened = prepare_query(tokenizer, query)
    assert shortened == query[:query.index(" q96")]
    assert len(encoded(tokenizer, shortened)["input_ids"]) == 96
    assert query.endswith("q139")


def test_condition_lead_in_and_exception_are_atomic_across_windows():
    text = ("Điều kiện chung phải đáp ứng:\n1. " + " ".join(["nguyên tắc"] * 400) +
            ".\n2. Trừ trường hợp khoản 1 có quy định khác.\n3. Quy định độc lập.")
    parent = chunk(0, text=text)
    windows = build_source_windows(parent, Tokenizer())
    groups = source_groups(parent, windows)
    related = next(group for group in groups if any(unit.khoan == "2" for unit in group.units))
    assert related.atomic and {u.khoan for u in related.units} == {None, "1", "2"}
    assert len(related.window_ids) >= 3
    assert all(u.text == text[u.start:u.end] for u in related.units)
    independent = next(group for group in groups if any(u.khoan == "3" for u in group.units))
    assert {u.khoan for u in independent.units} == {None, "3"}


def test_unstructured_source_is_one_complete_atomic_group():
    parent = chunk(0, text=" ".join(["điều kiện trừ trường hợp"] * 300))
    windows = build_source_windows(parent, Tokenizer())
    groups = source_groups(parent, windows)
    assert len(groups) == 1 and len(groups[0].window_ids) > 1
    assert groups[0].units[0].text == parent["text"]


@pytest.mark.parametrize("first,last", [("1", "1"), ("1dd", "1đ"), ("1đ", "1dd")])
def test_duplicate_normalized_clause_numbers_keep_whole_parent_atomic(first, last):
    text = (f"Điều kiện mở đầu:\n{first}. Điều kiện gốc " + " ".join(["chung"] * 700) +
            f".\n2. Áp dụng theo khoản {first} Điều này.\n{last}. Trích dẫn nội dung sửa đổi.")
    parent = chunk(0, text=text)
    windows = build_source_windows(parent, Tokenizer())
    groups = source_groups(parent, windows)
    assert len(groups) == 1 and groups[0].atomic
    assert len(groups[0].units) == 4
    assert [unit.khoan for unit in groups[0].units] == [None, first.replace("dd", "đ"), "2", last.replace("dd", "đ")]
    assert groups[0].window_ids == tuple(window.window_id for window in windows)
    assert len(groups[0].window_ids) > 1
    assert all(unit.text == text[unit.start:unit.end] for unit in groups[0].units)
    assert "".join(unit.text for unit in groups[0].units) == text


@pytest.mark.parametrize("clause,reference_id,normalized", [("1", "1", "1"),
    ("1dd", "1đ", "1đ"), ("1đ", "1dd", "1đ")])
def test_unique_clauses_keep_structural_reference_groups_separate(clause, reference_id, normalized):
    text = f"Điều kiện:\n{clause}. Điều kiện gốc.\n2. Áp dụng theo khoản {reference_id} Điều này.\n3. Quy định độc lập."
    parent = chunk(0, text=text)
    groups = source_groups(parent, build_source_windows(parent, Tokenizer()))
    assert [{unit.khoan for unit in group.units} for group in groups] == [{None, normalized, "2"}, {None, "3"}]


@pytest.mark.parametrize("bad", [None, lambda text, **kwargs: {"input_ids": [1]},
    lambda text, **kwargs: {"input_ids": [1], "offset_mapping": [(2, 1)]},
    lambda text, **kwargs: {"input_ids": [[1]], "offset_mapping": [(0, 1)]}])
def test_unsupported_tokenizer_has_explicit_reason(bad):
    with pytest.raises(UnsupportedTokenizerError) as raised:
        build_source_windows(chunk(0), bad)
    assert raised.value.reason


def test_rerank_long_parent_nominates_end_instead_of_blind_first():
    parent = chunk(0, text=" ".join(["chung"] * 1600) + " quyềntarget")
    instance = retriever([parent], raw=-2.75)
    result = instance.rerank_pool(pool_candidates([batch(instance, "quyềntarget")]), remaining_pairs=1)
    assert result.scored[0].nomination.window.end == len(parent["text"])
    assert result.scored[0].raw_score == -2.75
    assert result.scored[0].semantic_eligible is False
    assert result.spent_pairs == 1 and result.budget_exhausted
    assert len(result.unseen) >= 4 and result.unseen_parent_ids == ("c0",)


def test_coverage_tie_order_includes_end_before_middle():
    instance = retriever([chunk(0, text=" ".join(["chung"] * 1600))])
    result = instance.rerank_pool(pool_candidates([batch(instance, "khác")]), remaining_pairs=2)
    assert result.scored[0].nomination.window.start == 0
    assert result.scored[1].nomination.window.end == len(instance.chunks[0]["text"])


def test_special_tokens_count_and_complete_pair_cap():
    instance = retriever([chunk(0, text=" ".join(["passage"] * 384))])
    query = " ".join(["query"] * 120)
    pool = pool_candidates([batch(instance, query)])
    result = instance.rerank_pool(pool, tokenizer=Tokenizer(specials=32))
    assert result.spent_pairs == 1
    sent_query, sent_text = instance.reranker.predict.call_args.args[0][0]
    assert pair_token_count(Tokenizer(32), sent_query, sent_text) == 512
    blocked = instance.rerank_pool(pool, tokenizer=Tokenizer(specials=33))
    assert blocked.spent_pairs == 0 and len(blocked.unseen) == 1
    assert "pair_token_limit_exceeded" in blocked.warnings


def test_missing_offsets_return_unseen_not_blind_fallback():
    instance = retriever([chunk(0)])
    bad = lambda text, **kwargs: {"input_ids": [1]}
    result = instance.rerank_pool(pool_candidates([batch(instance)]), tokenizer=bad)
    assert result.spent_pairs == 0 and result.unseen_parent_ids == ("c0",)
    assert result.warnings == ("tokenizer_offsets_unsupported",)
    instance.reranker.predict.assert_not_called()


def test_raw_positive_negative_scores_and_direct_low_score_survive():
    question = "Điều 1"
    instance = retriever([chunk(0)], raw=-10)
    result = instance.rerank_pool(pool_candidates([batch(instance, question, references=[reference(question)])]))
    assert len(result.scored) == 1 and result.scored[0].raw_score == -10
    assert result.scored[0].nomination.direct and not result.scored[0].semantic_eligible
    instance.reranker.predict.side_effect = lambda pairs: np.full(len(pairs), 7.2)
    positive = instance.rerank_pool(pool_candidates([batch(instance)]))
    assert positive.scored[0].raw_score == 7.2 and positive.scored[0].semantic_eligible


def test_rerank_fair_across_issues_queries_and_parents():
    instance = retriever([chunk(0, text=" ".join(["nguồn"] * 1500)), chunk(1)])
    batches = [batch(instance, query_id="q1", issue_id="i1"),
               batch(instance, query_id="q2", issue_id="i1"),
               batch(instance, query_id="q3", issue_id="i2")]
    result = instance.rerank_pool(pool_candidates(batches), remaining_pairs=4)
    assert [(s.nomination.issue_id, s.nomination.query_id, s.nomination.window.parent_chunk_id)
            for s in result.scored] == [("i1", "q1", "c0"), ("i2", "q3", "c0"),
                                       ("i1", "q2", "c0"), ("i2", "q3", "c1")]


def test_request_local_72_plus_24_pair_budget_and_lower_remaining():
    instance = retriever([chunk(index, text=" ".join(["nguồn"] * 2500)) for index in range(12)])
    pool = pool_candidates([batch(instance)])
    initial = instance.rerank_pool(pool)
    assert initial.spent_pairs == 72 and initial.initial_pairs_after == 72
    supplement = instance.rerank_pool(pool, supplementary=True, initial_pairs_used=72)
    assert supplement.spent_pairs == 24 and supplement.supplementary_pairs_after == 24
    exhausted = instance.rerank_pool(pool, supplementary=True, initial_pairs_used=72, supplementary_pairs_used=24)
    assert exhausted.spent_pairs == 0 and exhausted.budget_exhausted
    lowered = instance.rerank_pool(pool, remaining_pairs=3)
    assert lowered.spent_pairs == 3 and lowered.initial_pairs_after == 3
    fresh = instance.rerank_pool(pool, remaining_pairs=1)
    assert fresh.spent_pairs == 1 and fresh.initial_pairs_after == 1


@pytest.mark.parametrize("kwargs", [{"initial_capacity": 73}, {"supplementary_capacity": 25},
    {"initial_pairs_used": -1}, {"initial_pairs_used": 73}, {"remaining_pairs": -1},
    {"remaining_pairs": True}, {"query_tokens": 97}, {"passage_tokens": 385},
    {"overlap_tokens": 384}, {"pair_tokens": 513}])
def test_invalid_rerank_capacity_rejected(kwargs):
    instance = retriever([chunk(0)])
    with pytest.raises(ValueError):
        instance.rerank_pool(pool_candidates([batch(instance)]), **kwargs)
    instance.reranker.predict.assert_not_called()


def test_reranker_bad_scores_are_not_silently_normalized():
    instance = retriever([chunk(0)])
    instance.reranker.predict.side_effect = lambda pairs: [float("nan")]
    with pytest.raises(ValueError, match="raw scores"):
        instance.rerank_pool(pool_candidates([batch(instance)]))


def test_direct_point_window_first_and_condition_lead_in_retained():
    question = "điểm b khoản 1 Điều 1"
    text = ("Điều kiện dẫn nhập:\n1. a) " + " ".join(["chung"] * 900) +
            "\nb) Đích rõ ràng.\n2. Trừ trường hợp khoản 1 có quy định khác.")
    instance = retriever([chunk(0, text=text)])
    pool = pool_candidates([batch(instance, question, references=[reference(question, khoan="1", diem="b")])])
    result = instance.rerank_pool(pool, remaining_pairs=2)
    assert result.scored[0].nomination.targeted
    assert "Đích rõ ràng" in result.scored[0].nomination.window.text
    assert result.scored[1].nomination.window.start == 0
    assert result.groups[0].units[0].start == 0
    assert {u.khoan for u in result.groups[0].units} == {None, "1", "2"}


def test_active_pool_48_parent_cap_and_fair_overflow():
    instance = retriever([chunk(index) for index in range(72)])
    batches = []
    for issue in range(6):
        values = np.zeros(72)
        values[issue * 12:(issue + 1) * 12] = np.arange(12, 0, -1) + 100
        instance.emb = values.reshape(-1, 1)
        instance.bm25.get_scores.return_value = values
        instance.bm25_noacc.get_scores.return_value = values
        batches.append(batch(instance, query_id=f"q{issue}", issue_id=f"i{issue}"))
    result = pool_candidates(batches)
    assert len(result.parents) == 48 and len(result.excluded_parent_ids) == 24
    assert {issue: sum(any(c.issue_id == issue for c in p.nominations) for p in result.parents)
            for issue in [f"i{i}" for i in range(6)]} == dict.fromkeys([f"i{i}" for i in range(6)], 8)


def test_shared_parent_targeted_direct_beats_earlier_semantic_at_last_pair():
    question = "khoản 2 Điều 1"
    text = "1. " + " ".join(["chung"] * 1500) + ".\n2. Điều kiện đích cuối nguồn."
    instance = retriever([chunk(0, text=text)])
    semantic = batch(instance, "chung", query_id="q-semantic", issue_id="i-semantic",
                     original_question=question)
    direct = batch(instance, question, query_id="q-direct", issue_id="i-direct",
                   references=[reference(question, khoan="2")])
    pool = pool_candidates([semantic, direct], max_parents=1)
    assert [c.direct for c in pool.parents[0].nominations] == [False, True]
    result = instance.rerank_pool(pool, remaining_pairs=1)
    winner = result.scored[0].nomination
    assert winner.issue_id == "i-direct" and winner.targeted
    assert winner.window.end == len(text)
    assert any(item.issue_id == "i-semantic" and item.window.start == 0 for item in result.unseen)


def test_target_priority_is_fair_between_issues_with_same_priority():
    question = "khoản 2 Điều 1"
    text = "1. " + " ".join(["chung"] * 700) + ".\n2. " + " ".join(["đích"] * 800)
    instance = retriever([chunk(0, text=text)])
    batches = [batch(instance, question, query_id=f"q{i}", issue_id=f"i{i}",
                     references=[reference(question, khoan="2")]) for i in range(2)]
    result = instance.rerank_pool(pool_candidates(batches), remaining_pairs=2)
    assert [item.nomination.issue_id for item in result.scored] == ["i0", "i1"]
    assert all(item.nomination.targeted for item in result.scored)


def test_supplementary_exclusions_open_unseen_without_rescoring_old_pairs():
    instance = retriever([chunk(0, text=" ".join(["nguồn"] * 1800))])
    pool = pool_candidates([batch(instance)])
    initial = instance.rerank_pool(pool, remaining_pairs=2)
    prior_keys = {(s.nomination.issue_id, s.nomination.query_id, s.nomination.window.window_id)
                  for s in initial.scored}
    supplemental = instance.rerank_pool(pool, supplementary=True,
        initial_pairs_used=initial.initial_pairs_after, remaining_pairs=2, exclude_scored=prior_keys)
    next_keys = {(s.nomination.issue_id, s.nomination.query_id, s.nomination.window.window_id)
                 for s in supplemental.scored}
    unseen_keys = {(s.issue_id, s.query_id, s.window.window_id) for s in supplemental.unseen}
    assert prior_keys.isdisjoint(next_keys | unseen_keys)
    assert len(next_keys) == 2 and supplemental.spent_pairs == 2
    assert supplemental.initial_pairs_after == 2 and supplemental.supplementary_pairs_after == 2
    original_unseen = {s.window.window_id for s in initial.unseen}
    assert {s.nomination.window.window_id for s in supplemental.scored} <= original_unseen
    all_seen = {(item.issue_id, item.query_id, item.window.window_id)
                for item in (*[s.nomination for s in initial.scored], *initial.unseen)}
    exhausted_views = instance.rerank_pool(pool, supplementary=True, exclude_scored=all_seen)
    assert exhausted_views.spent_pairs == 0 and not exhausted_views.unseen
    assert not exhausted_views.unseen_parent_ids and not exhausted_views.budget_exhausted


@pytest.mark.parametrize("key", [("i1", "q1"), ("i1", "q1", ""), "i1q1window"])
def test_invalid_exclusion_key_is_rejected(key):
    instance = retriever([chunk(0)])
    with pytest.raises(ValueError, match="exclude_scored"):
        instance.rerank_pool(pool_candidates([batch(instance)]), exclude_scored=[key])


@pytest.mark.parametrize("kwargs", [{"per_channel": 21}, {"per_query": 13},
                                  {"per_channel": 0}, {"per_query": True}])
def test_invalid_candidate_limits_rejected(kwargs):
    instance = retriever([chunk(0)])
    with pytest.raises(ValueError):
        batch(instance, **kwargs)
    instance.embedder.encode.assert_not_called()


def main():
    return pytest.main([__file__, *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
