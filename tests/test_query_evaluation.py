"""Pure offline evaluation contracts; no model/runtime imports or requests."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from copy import deepcopy
import json
import subprocess

import pytest

from ragdn.query_evaluation import (
    aggregate_case_reports, corpus_semantic_hash, evaluate_case, evaluate_suite,
    evidence_coverage, quantiles, retrieval_at_k, summarize_observations,
    validate_case_annotation, validate_smoke_artifact,
)


def fixture_data():
    text = "1. Lead-in condition applies.\n2. Except on Tuesday.\n3. Independent rule.\n"
    chunks = [{"chunk_id": "p1", "doc_id": "law", "dieu": "1", "text": text, "hieu_luc": True},
              {"chunk_id": "p2", "doc_id": "other", "dieu": "1", "text": "Alternative complete rule.", "hieu_luc": False}]
    def span(parent, start=0, end=None, **fields):
        end = len(parent["text"]) if end is None else end
        return {"parent_chunk_id": parent["chunk_id"], "doc_id": parent["doc_id"],
                "dieu": parent["dieu"], "start": start, "end": end,
                "quote": parent["text"][start:end], **fields}
    first_end = text.index("2.")
    second_end = text.index("3.")
    lead = span(chunks[0], 0, first_end, khoan="1")
    exception = span(chunks[0], first_end, second_end, khoan="2")
    alternative = span(chunks[1])
    case = {"case_id": "T1", "corpus_sha256": corpus_semantic_hash(chunks),
            "assessment_status": "approved", "expected_articles": ["law:1"],
            "issues": [{"issue_id": "i1", "assessment_status": "approved", "requirements": [
                {"requirement_id": "r1", "assessment_status": "approved", "alternatives": [
                    {"group_id": "joint", "spans": [lead, exception]},
                    {"group_id": "alternative", "spans": [alternative]}]}]}]}
    return chunks, case, lead, exception, alternative


def artifact_for(case):
    return {"schema_version": 1, "annotation_status": "approved",
            "corpus_sha256": case["corpus_sha256"], "cases": [case]}


def test_doc_article_metrics_deduplicate_before_same_k_as_legacy():
    result = retrieval_at_k(["wrong:1", "wrong:1", "law:1", "law:2"], ["law:1", "law:2"], k=2)
    assert result["hit"] == 1 and result["recall"] == .5 and result["mrr"] == .5
    assert result["retrieved_articles"] == 2
    miss = retrieval_at_k(["wrong:1", "law:1"], ["law:1"], k=1)
    assert (miss["hit"], miss["recall"], miss["mrr"]) == (0, 0, 0)


def test_same_article_number_in_another_document_is_not_a_hit():
    result = retrieval_at_k([{"doc_id": "other", "dieu": "1"}], ["law:1"])
    assert result["hit"] == result["recall"] == result["mrr"] == 0


@pytest.mark.parametrize("retrieved,expected", [(None, ["law:1"]), ([], None), ([], [])])
def test_missing_expected_or_retrieval_is_unassessed(retrieved, expected):
    result = retrieval_at_k(retrieved, expected)
    assert result["status"] == "not_assessed" and result["hit"] is None


def test_explicit_empty_retrieval_is_assessed_miss_with_approved_expected_set():
    result = retrieval_at_k([], ["law:1"])
    assert result["status"] == "assessed" and result["recall"] == 0


def test_all_spans_in_one_alternative_must_be_present_together():
    chunks, case, lead, exception, _ = fixture_data()
    partial = evidence_coverage(case, [lead], chunks)
    assert partial["status"] == "assessed" and partial["all_required_covered"] is False
    complete = evidence_coverage(case, [lead, exception], chunks)
    assert complete["all_required_covered"] is True


def test_one_complete_alternative_can_satisfy_requirement():
    chunks, case, _, _, alternative = fixture_data()
    assert evidence_coverage(case, [alternative], chunks)["all_required_covered"] is True


def test_two_requirements_are_conjunctive_even_with_alternatives():
    chunks, case, lead, exception, alternative = fixture_data()
    case["issues"][0]["requirements"].append({"requirement_id": "r2", "assessment_status": "approved",
        "alternatives": [{"group_id": "second", "spans": [alternative]}]})
    assert evidence_coverage(case, [lead, exception], chunks)["all_required_covered"] is False
    assert evidence_coverage(case, [lead, exception, alternative], chunks)["all_required_covered"] is True


def test_adjacent_actual_context_spans_can_jointly_cover_required_span():
    chunks, case, lead, exception, _ = fixture_data()
    split = lead["start"] + 8
    left = {**lead, "end": split, "quote": chunks[0]["text"][lead["start"]:split]}
    right = {**lead, "start": split, "quote": chunks[0]["text"][split:lead["end"]]}
    assert evidence_coverage(case, [left, right, exception], chunks)["all_required_covered"] is True
    gap = {**right, "start": split + 1, "quote": chunks[0]["text"][split + 1:lead["end"]]}
    assert evidence_coverage(case, [left, gap, exception], chunks)["all_required_covered"] is False


def test_article_identity_alone_never_satisfies_actual_context_coverage():
    chunks, case, _, _, _ = fixture_data()
    result = evaluate_case(case, {"retrieved_articles": ["law:1"]}, chunks)
    assert result["retrieval"]["hit"] == 1
    assert result["coverage"]["status"] == "not_assessed"
    assert result["answer_quality"]["status"] == "not_assessed"


@pytest.mark.parametrize("field,value", [("doc_id", "other"), ("dieu", "99"),
    ("parent_chunk_id", "absent"), ("start", -1), ("start", True),
    ("end", 10000), ("quote", "invented"), ("khoan", "2")])
def test_invalid_gold_locator_cannot_produce_successful_metrics(field, value):
    chunks, case, lead, exception, _ = fixture_data()
    case["issues"][0]["requirements"][0]["alternatives"][0]["spans"][0][field] = value
    result = evaluate_case(case, {"retrieved_articles": ["law:1"], "context_spans": [lead, exception]}, chunks)
    assert result["annotation"]["status"] == "invalid"
    assert result["retrieval"]["status"] == result["coverage"]["status"] == "not_assessed"


def test_wrong_actual_context_document_is_not_counted_as_covered():
    chunks, case, lead, exception, _ = fixture_data()
    bad = {**lead, "doc_id": "other"}
    result = evidence_coverage(case, [bad, exception], chunks)
    assert result["status"] == "not_assessed" and result["reason"] == "invalid_actual_context_span"


def test_stale_semantic_hash_invalidates_all_annotation_metrics():
    chunks, case, lead, exception, _ = fixture_data()
    changed = deepcopy(chunks)
    changed[0]["text"] += "new source bytes"
    result = evaluate_suite(artifact_for(case), {"T1": {"retrieved_articles": ["law:1"],
                             "context_spans": [lead, exception]}}, changed)
    assert result["status"] == "not_assessed" and result["cases"] == []
    assert result["annotation"]["errors"] == ["stale_corpus_hash"]


def test_one_stale_case_hash_invalidates_suite_even_if_top_hash_matches():
    chunks, case, _, _, _ = fixture_data()
    artifact = artifact_for(case)
    case["corpus_sha256"] = "stale"
    artifact["corpus_sha256"] = corpus_semantic_hash(chunks)
    assert evaluate_suite(artifact, {}, chunks)["status"] == "not_assessed"


@pytest.mark.parametrize("level", ["case", "issue", "requirement"])
def test_draft_requirements_never_become_official_coverage_pass_or_failure(level):
    chunks, case, lead, exception, _ = fixture_data()
    node = case if level == "case" else case["issues"][0] if level == "issue" else case["issues"][0]["requirements"][0]
    node["assessment_status"] = "needs_user_review"
    result = evidence_coverage(case, [lead, exception], chunks)
    assert result["status"] == "not_assessed" and result["all_required_covered"] is None
    diagnostic = result["issues"]["i1"]["draft_diagnostics"]
    assert diagnostic[0]["covered"] is True


def test_draft_case_blocks_retrieval_and_answer_quality_rates_even_with_judgments():
    chunks, case, lead, exception, _ = fixture_data()
    case["assessment_status"] = "needs_user_review"
    result = evaluate_suite(artifact_for(case), {"T1": {"retrieved_articles": ["law:1"],
        "context_spans": [lead, exception], "answer_judgments": {"false_supported": False}}}, chunks)
    assert result["retrieval"]["status"] == result["answer_quality"]["false_supported"]["status"] == "not_assessed"
    assert result["evidence_coverage"]["value"] is None
    assert result["annotation_statuses"] == ["needs_user_review"]
    assert result["acceptance"] == "user_review_required"


def test_one_unannotated_issue_prevents_whole_case_success():
    chunks, case, lead, exception, _ = fixture_data()
    case["issues"].append({"issue_id": "i2", "assessment_status": "not_assessed", "requirements": []})
    result = evidence_coverage(case, [lead, exception], chunks)
    assert result["status"] == "not_assessed" and result["all_required_covered"] is None
    assert result["issues"]["i1"]["all_required_covered"] is True
    assert result["issues"]["i2"]["all_required_covered"] is None


def test_missing_annotation_is_not_an_empty_set_pass():
    chunks, case, _, _, _ = fixture_data()
    case["issues"] = []
    result = evaluate_case(case, {"retrieved_articles": []}, chunks)
    assert result["annotation"]["status"] == "not_assessed"
    assert result["retrieval"]["hit"] is None


def test_answer_error_rates_require_supplied_judgments_and_keep_unknown_denominators():
    chunks, case, lead, exception, _ = fixture_data()
    one = evaluate_case(case, {"retrieved_articles": ["law:1"], "context_spans": [lead, exception],
        "answer_judgments": {"false_supported": True, "wrong_document": True,
            "repealed_as_current": False, "hard_negative_used": False, "factual_preservation": True}}, chunks)
    two = evaluate_case(case, {"retrieved_articles": []}, chunks)
    summary = aggregate_case_reports([one, two])
    assert summary["retrieval"]["hit"] == .5
    assert summary["answer_quality"]["false_supported"] == {
        "status": "assessed", "n": 1, "not_assessed": 1, "value": 1.0}
    assert summary["answer_quality"]["repealed_as_current"]["value"] == 0
    assert summary["evidence_coverage"]["n"] == 1 and summary["evidence_coverage"]["not_assessed"] == 1


def test_successful_retrieval_does_not_infer_answer_correctness():
    chunks, case, lead, exception, _ = fixture_data()
    result = evaluate_case(case, {"retrieved_articles": ["law:1"], "context_spans": [lead, exception]}, chunks)
    assert result["coverage"]["all_required_covered"] is True
    assert result["answer_quality"]["status"] == "not_assessed"
    assert result["answer_quality"]["false_supported"] is None


def test_linear_quantiles_have_explicit_missing_and_singleton_behavior():
    assert quantiles([])["p95"] is None
    assert quantiles([2])["p50"] == quantiles([2])["p95"] == 2
    result = quantiles([1, 2, 3, 4, 5])
    assert result["p50"] == 3 and result["p95"] == pytest.approx(4.8)


@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf"), "2"])
def test_invalid_measurements_are_rejected(value):
    with pytest.raises(ValueError):
        quantiles([value])


def test_operational_rates_counts_and_stage_times_do_not_impute_missing_as_zero():
    result = summarize_observations([
        {"fallback": True, "invalid_json": False, "budget_exhausted": True,
         "completion_calls": 3, "rerank_pairs": 96, "context_tokens": 700,
         "stage_seconds": {"planner": 1, "mapper": 3}},
        {"fallback": False, "completion_calls": 1, "rerank_pairs": 2,
         "stage_seconds": {"planner": 5}}, {}])
    assert result["operational_rates"]["fallback"]["value"] == .5
    assert result["operational_rates"]["invalid_json"]["not_assessed"] == 2
    assert result["counters"]["rerank_pairs"]["total"] == 98
    assert result["counters"]["context_tokens"]["not_assessed"] == 2
    assert result["stage_seconds"]["stages"]["planner"]["p50"] == 3
    assert result["stage_seconds"]["stages"]["mapper"]["not_assessed"] == 2
    empty = summarize_observations([{}])
    assert empty["counters"]["completion_calls"]["total"] is None
    assert empty["operational_rates"]["invalid_json"]["value"] is None


def test_mixed_k_reports_cannot_be_compared():
    chunks, case, _, _, _ = fixture_data()
    one = evaluate_case(case, {"retrieved_articles": []}, chunks, k=1)
    two = evaluate_case(case, {"retrieved_articles": []}, chunks, k=2)
    with pytest.raises(ValueError, match="same k"):
        aggregate_case_reports([one, two])


def test_corpus_hash_is_order_sensitive_and_matches_canonical_index_encoding():
    chunks, _, _, _, _ = fixture_data()
    assert corpus_semantic_hash(chunks) != corpus_semantic_hash(list(reversed(chunks)))
    import hashlib
    expected = hashlib.sha256(json.dumps([(c["chunk_id"], c["text"]) for c in chunks],
        ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert corpus_semantic_hash(chunks) == expected


def test_real_24_case_artifact_is_draft_exact_and_independent_of_runtime_prompts():
    artifact = json.loads((ROOT / "eval/query_pipeline_smoke.json").read_text(encoding="utf-8"))
    chunks = [json.loads(line) for line in (ROOT / "data/processed/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    golden = {row["id"]: row for row in (json.loads(line) for line in
        (ROOT / "eval/golden_dev_test_retrieval.jsonl").read_text(encoding="utf-8").splitlines())}
    assert len(artifact["cases"]) == 24
    original_ids = {"Q015", "Q055", "Q069", "Q079", "Q081", "Q083", "Q087", "Q091", "Q093", "Q096", "Q097", "Q099"}
    copied = [case for case in artifact["cases"] if case["origin"] == "existing_golden_copy"]
    assert {case["case_id"] for case in copied} == original_ids
    assert all(case["question"] == golden[case["case_id"]]["question"] for case in copied)
    assert len({case["category"] for case in artifact["cases"] if case["origin"] == "synthetic"}) == 12
    validation = validate_smoke_artifact(artifact, chunks)
    assert validation["status"] == "valid"
    assert sum(row["locator_count"] for row in validation["cases"].values()) > 24
    assert all(case["assessment_status"] == "needs_user_review" for case in artifact["cases"])
    report = evaluate_suite(artifact, {}, chunks)
    assert report["retrieval"]["n"] == 0 and report["retrieval"]["hit"] is None
    assert report["evidence_coverage"]["n"] == 0 and report["evidence_coverage"]["value"] is None
    serialized = json.dumps(report)
    assert all(case["question"] not in serialized for case in artifact["cases"])
    assert "quote" not in serialized


def test_import_does_not_load_retrieval_index_pipeline_or_model_libraries():
    code = ("import sys; sys.path.insert(0, " + repr(str(ROOT / "src")) + "); "
            "import ragdn.query_evaluation; "
            "assert not any(name in sys.modules for name in "
            "('ragdn.retrieval','ragdn.index','ragdn.pipeline','sentence_transformers','torch','openai'))")
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_diem_locator_supports_inline_points_and_rejects_wrong_or_ambiguous_scope():
    text = "1. Điều kiện: a) Điểm đầu; c1) Điểm bổ sung; đ) Điểm cuối.\n2. Ngoại lệ: a) Điểm trùng nhãn.\n"
    chunks = [{"chunk_id": "p1", "doc_id": "law", "dieu": "1", "text": text, "hieu_luc": True}]
    c1_start, c1_end = text.index("c1)"), text.index("đ)")
    dd_start, dd_end = text.index("đ)"), text.index("2.")
    a1_start, a1_end = text.index("a)"), text.index("c1)")
    base = {"parent_chunk_id": "p1", "doc_id": "law", "dieu": "1"}
    valid_spans = [
        {**base, "khoan": "1", "diem": "c1", "start": c1_start, "end": c1_end, "quote": text[c1_start:c1_end]},
        {**base, "khoan": "1", "diem": "dd", "start": dd_start, "end": dd_end, "quote": text[dd_start:dd_end]},
    ]
    case = {"case_id": "D1", "corpus_sha256": corpus_semantic_hash(chunks), "assessment_status": "approved",
            "issues": [{"issue_id": "i1", "assessment_status": "approved", "requirements": [
                {"requirement_id": "r1", "assessment_status": "approved",
                 "alternatives": [{"group_id": "g1", "spans": valid_spans}]}]}]}
    assert validate_case_annotation(case, chunks)["status"] == "valid"
    wrong_diem = deepcopy(case)
    wrong_diem["issues"][0]["requirements"][0]["alternatives"][0]["spans"][0]["diem"] = "b"
    assert validate_case_annotation(wrong_diem, chunks)["status"] == "invalid"
    ambiguous_unscoped = deepcopy(case)
    ambiguous_unscoped["issues"][0]["requirements"][0]["alternatives"][0]["spans"] = [
        {**base, "diem": "a", "start": a1_start, "end": a1_end, "quote": text[a1_start:a1_end]}]
    assert validate_case_annotation(ambiguous_unscoped, chunks)["status"] == "invalid"


@pytest.mark.parametrize("bad_alternatives", [
    None, "malformed", {}, [{}], [{"spans": "not_a_list"}], [{"spans": []}],
    [{"spans": [{"parent_chunk_id": "missing"}]}],
])
def test_malformed_draft_alternatives_never_crash_or_become_assessed_passes(bad_alternatives):
    chunks, case, lead, exception, _ = fixture_data()
    case["assessment_status"] = "needs_user_review"
    case["issues"][0]["assessment_status"] = "needs_user_review"
    case["issues"][0]["requirements"][0]["assessment_status"] = "needs_user_review"
    case["issues"][0]["requirements"][0]["alternatives"] = bad_alternatives
    cov = evidence_coverage(case, [lead, exception], chunks)
    assert cov["status"] != "assessed" and cov.get("all_required_covered") is not True
    rep = evaluate_case(case, {"retrieved_articles": ["law:1"], "context_spans": [lead, exception]}, chunks)
    assert rep["coverage"]["status"] != "assessed" and rep["retrieval"]["status"] == "not_assessed"


def test_null_and_missing_stage_measurements_and_none_observations_stay_unassessed():
    summary = summarize_observations([
        None,
        {"stage_seconds": None},
        {"stage_seconds": {"planner": None, "mapper": 2.0}},
    ])
    assert summary["stage_seconds"]["status"] == "assessed"
    assert summary["stage_seconds"]["stages"]["planner"] == {
        "status": "not_assessed", "reason": "missing_measurements",
        "n": 0, "p50": None, "p95": None, "not_assessed": 3,
    }
    assert summary["stage_seconds"]["stages"]["mapper"]["n"] == 1
    assert summary["stage_seconds"]["stages"]["mapper"]["not_assessed"] == 2
    all_null = summarize_observations([None, {"stage_seconds": {"planner": None}}])
    assert all_null["stage_seconds"]["status"] == "not_assessed"
    for bad_obs in ([{"stage_seconds": []}], [{"stage_seconds": {"planner": -1}}],
                    [{"completion_calls": True}], [{"fallback": 1}]):
        with pytest.raises(ValueError):
            summarize_observations(bad_obs)


def test_invalid_inputs_and_expected_articles_are_rejected_without_false_certainty():
    chunks, case, _, _, _ = fixture_data()
    with pytest.raises(ValueError):
        retrieval_at_k("law:1", ["law:1"])
    with pytest.raises(ValueError):
        retrieval_at_k([" :1"], ["law:1"])
    with pytest.raises(ValueError):
        retrieval_at_k(["law:1"], ["law:1"], k=0)
    bad_expected = deepcopy(case)
    bad_expected["expected_articles"] = ["invalid_identity"]
    assert validate_case_annotation(bad_expected, chunks)["status"] == "invalid"
    cov = evidence_coverage(case, "not_a_span_list", chunks)
    assert cov["status"] == "not_assessed" and cov["reason"] == "invalid_actual_context_span"


@pytest.mark.parametrize("bad_parent_id", ["", "   ", None, 123, [], {}])
def test_invalid_or_unhashable_parent_chunk_id_rejected_in_annotations_and_context_spans(bad_parent_id):
    chunks, approved_case, lead, exception, _ = fixture_data()
    draft_case = deepcopy(approved_case)
    draft_case["assessment_status"] = "needs_user_review"
    assert evidence_coverage(approved_case, [lead, exception], chunks)["all_required_covered"] is True
    draft_ok = evidence_coverage(draft_case, [lead, exception], chunks)
    assert draft_ok["status"] == "not_assessed" and draft_ok["issues"]["i1"]["draft_diagnostics"][0]["covered"] is True
    for base_case in (approved_case, draft_case):
        bad_ann = deepcopy(base_case)
        bad_ann["issues"][0]["requirements"][0]["alternatives"][0]["spans"][0]["parent_chunk_id"] = bad_parent_id
        ann_res = validate_case_annotation(bad_ann, chunks)
        assert ann_res["status"] == "invalid" and "invalid_source_locator" in ann_res["errors"]
        case_rep = evaluate_case(bad_ann, {"retrieved_articles": ["law:1"], "context_spans": [lead, exception]}, chunks)
        assert case_rep["annotation"]["status"] == "invalid" and case_rep["coverage"]["status"] == "not_assessed"
        bad_ctx = evidence_coverage(base_case, [{**lead, "parent_chunk_id": bad_parent_id}, exception], chunks)
        assert bad_ctx["status"] == "not_assessed" and bad_ctx["reason"] == "invalid_actual_context_span"


def main():
    return pytest.main([__file__, *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
