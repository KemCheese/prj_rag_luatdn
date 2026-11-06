"""Eval-only pure helpers. Never import this module into the query runtime.

Inputs are captured observations and separately reviewed annotations. These
functions never read files, construct models, retrieve, send prompts or judge
legal correctness. Missing judgments and draft annotations remain unassessed.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence


def corpus_semantic_hash(chunks: Sequence[Mapping]) -> str:
    """Match index.chunks_sha256 without importing retrieval/tokenizer code."""
    payload = json.dumps([(row["chunk_id"], row["text"]) for row in chunks],
                         ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _unassessed(reason: str, **fields) -> dict:
    return {"status": "not_assessed", "reason": reason, **fields}


def _identity(value) -> tuple[str, str]:
    if isinstance(value, str) and ":" in value:
        doc, article = value.split(":", 1)
    elif isinstance(value, Mapping):
        doc, article = value.get("doc_id"), value.get("dieu")
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        doc, article = value
    else:
        raise ValueError("article identity requires doc_id and dieu")
    if not isinstance(doc, str) or not doc.strip() or not isinstance(article, str) or not article.strip():
        raise ValueError("article identity requires nonempty strings")
    return doc, article


def retrieval_at_k(retrieved, expected, k: int = 5) -> dict:
    """Deduplicate doc/article identities before k, matching legacy eval.py.

    This primitive assumes its expected set has already been approved/validated.
    evaluate_case enforces annotation review and corpus validity first.
    """
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if retrieved is None or expected is None or not expected:
        return _unassessed("missing_retrieval_or_expected_articles", k=k,
                           hit=None, recall=None, mrr=None)
    if isinstance(retrieved, (str, bytes, Mapping)) or isinstance(expected, (str, bytes, Mapping)):
        raise ValueError("retrieved and expected must be sequences of article identities")
    gold = {_identity(item) for item in expected}
    ranked = list(dict.fromkeys(_identity(item) for item in retrieved))[:k]
    found = [item in gold for item in ranked]
    return {"status": "assessed", "k": k, "hit": float(any(found)),
            "recall": len(gold.intersection(ranked)) / len(gold),
            "mrr": next((1 / rank for rank, match in enumerate(found, 1) if match), 0.0),
            "retrieved_articles": len(ranked), "expected_articles": len(gold)}


def _source_map(chunks):
    parents = {row["chunk_id"]: row for row in chunks}
    if len(parents) != len(chunks):
        raise ValueError("duplicate source parent IDs")
    return parents


def _valid_locator(locator, parents) -> bool:
    if not isinstance(locator, Mapping):
        return False
    parent_id = locator.get("parent_chunk_id")
    if not isinstance(parent_id, str) or not parent_id.strip():
        return False
    parent = parents.get(parent_id)
    if parent is None:
        return False
    start, end, quote = locator.get("start"), locator.get("end"), locator.get("quote")
    valid = (locator.get("doc_id") == parent["doc_id"] and locator.get("dieu") == parent["dieu"] and
            type(start) is int and type(end) is int and 0 <= start < end <= len(parent["text"]) and
            isinstance(quote, str) and bool(quote.strip()) and quote == parent["text"][start:end])
    if not valid:
        return False
    scope_start, scope_end = 0, len(parent["text"])
    for field, pattern in (("khoan", r"(?m)^\s*(\d+(?:dd|[a-zđ])?)\.\s+"),
                           ("diem", r"(?m)(?:^\s*|(?<=[:;.]\s))((?:dd|[a-zđ])\d*)\)\s+")):
        label = locator.get(field)
        if label is None:
            continue
        if not isinstance(label, str) or not label.strip():
            return False
        normalized = unicodedata.normalize("NFC", label.strip()).lower().replace("dd", "đ")
        matches = list(re.finditer(pattern, parent["text"][scope_start:scope_end]))
        ranges = [(scope_start + match.start(), scope_start + (matches[index + 1].start()
                   if index + 1 < len(matches) else scope_end - scope_start))
                  for index, match in enumerate(matches)
                  if unicodedata.normalize("NFC", match.group(1)).lower().replace("dd", "đ") == normalized]
        containing = [(left, right) for left, right in ranges if left <= start and end <= right]
        if len(ranges) != 1 or len(containing) != 1:
            return False
        scope_start, scope_end = containing[0]
    return True


def validate_case_annotation(case: Mapping, chunks: Sequence[Mapping]) -> dict:
    """Validate mechanical provenance only; draft semantics stay draft."""
    if not isinstance(case, Mapping):
        return {"status": "invalid", "errors": ["case_not_mapping"]}
    if case.get("corpus_sha256") != corpus_semantic_hash(chunks):
        return {"status": "invalid", "errors": ["stale_corpus_hash"]}
    parents = _source_map(chunks)
    errors, locator_count = [], 0
    if not isinstance(case.get("case_id"), str) or not case["case_id"]:
        errors.append("case_id_missing")
    review = case.get("assessment_status")
    if review not in {"approved", "needs_user_review", "not_assessed"}:
        errors.append("assessment_status_missing_or_invalid")
    expected_articles = case.get("expected_articles")
    if expected_articles is not None:
        if not isinstance(expected_articles, list):
            errors.append("expected_articles_not_list")
        else:
            for item in expected_articles:
                try:
                    _identity(item)
                except ValueError:
                    errors.append("invalid_expected_article")
    issues = case.get("issues")
    if not isinstance(issues, list) or not issues:
        return _unassessed("missing_issue_annotations", locator_count=0)
    issue_ids = set()
    for issue in issues:
        if not isinstance(issue, Mapping):
            errors.append("issue_not_mapping")
            continue
        issue_id = issue.get("issue_id")
        if not isinstance(issue_id, str) or not issue_id or issue_id in issue_ids:
            errors.append("issue_id_missing_or_duplicate")
        if isinstance(issue_id, str):
            issue_ids.add(issue_id)
        if issue.get("assessment_status") not in {"approved", "needs_user_review", "not_assessed"}:
            errors.append("issue_assessment_status_invalid")
        requirements = issue.get("requirements", [])
        if not isinstance(requirements, list):
            errors.append("requirements_not_list")
            continue
        for requirement in requirements:
            if not isinstance(requirement, Mapping):
                errors.append("requirement_not_mapping")
                continue
            alternatives = requirement.get("alternatives")
            if requirement.get("assessment_status") not in {"approved", "needs_user_review", "not_assessed"}:
                errors.append("requirement_assessment_status_invalid")
            if alternatives is not None and not isinstance(alternatives, list):
                errors.append("empty_or_invalid_alternative_group")
                continue
            if not alternatives:
                if requirement.get("assessment_status") == "approved":
                    errors.append("approved_requirement_has_no_alternatives")
                continue
            for group in alternatives:
                if not isinstance(group, Mapping) or not isinstance(group.get("spans"), list) or not group["spans"]:
                    errors.append("empty_or_invalid_alternative_group")
                    continue
                for locator in group["spans"]:
                    locator_count += 1
                    if not _valid_locator(locator, parents):
                        errors.append("invalid_source_locator")
    return {"status": "invalid" if errors else "valid", "errors": sorted(set(errors)),
            "assessment_status": review, "locator_count": locator_count}


def validate_smoke_artifact(artifact: Mapping, chunks: Sequence[Mapping]) -> dict:
    if not isinstance(artifact, Mapping):
        return {"status": "invalid", "errors": ["artifact_not_mapping"], "cases": {}}
    if artifact.get("schema_version") != 1:
        return {"status": "invalid", "errors": ["unsupported_schema_version"], "cases": {}}
    actual = corpus_semantic_hash(chunks)
    if artifact.get("corpus_sha256") != actual:
        return {"status": "invalid", "errors": ["stale_corpus_hash"], "cases": {}}
    cases = artifact.get("cases")
    if not isinstance(cases, list) or not cases:
        return _unassessed("missing_cases", cases={})
    results, duplicates = {}, False
    for case in cases:
        if not isinstance(case, Mapping) or not isinstance(case.get("case_id"), str):
            return {"status": "invalid", "errors": ["invalid_case_record"], "cases": results}
        case_id = case.get("case_id")
        if case_id in results:
            duplicates = True
        results[case_id] = validate_case_annotation(case, chunks)
    invalid = duplicates or any(row["status"] == "invalid" for row in results.values())
    return {"status": "invalid" if invalid else "valid", "cases": results,
            "errors": ["duplicate_case_ids"] if duplicates else [],
            "annotation_status": artifact.get("annotation_status", "not_assessed")}


def _contains_span(required, supplied) -> bool:
    intervals = sorted((row["start"], row["end"]) for row in supplied
                       if row["parent_chunk_id"] == required["parent_chunk_id"])
    cursor = required["start"]
    for start, end in intervals:
        if start > cursor:
            break
        if end > cursor:
            cursor = end
        if cursor >= required["end"]:
            return True
    return False


def evidence_coverage(case: Mapping, context_spans, chunks: Sequence[Mapping]) -> dict:
    """All requirements AND; alternative groups OR; spans in one group AND.

    Context spans must describe text actually supplied to the mapper/reader, not
    merely retrieved parent IDs. Adjacent valid spans may jointly cover a range.
    Draft requirements produce diagnostics only, never official passes/failures.
    """
    validation = validate_case_annotation(case, chunks)
    if validation["status"] != "valid":
        return _unassessed("invalid_or_missing_annotation", annotation=validation,
                           all_required_covered=None, issues={})
    if context_spans is None:
        return _unassessed("missing_actual_context_spans", all_required_covered=None, issues={})
    if not isinstance(context_spans, Sequence) or isinstance(context_spans, (str, bytes, Mapping)):
        return _unassessed("invalid_actual_context_span", all_required_covered=None, issues={})
    parents = _source_map(chunks)
    if any(not _valid_locator(span, parents) for span in context_spans):
        return _unassessed("invalid_actual_context_span", all_required_covered=None, issues={})
    issues = {}
    approved_case = case.get("assessment_status") == "approved"
    for issue in case["issues"]:
        requirements = issue.get("requirements", [])
        diagnostic = []
        for requirement in requirements:
            groups = requirement.get("alternatives") or []
            covered = (any(all(_contains_span(span, context_spans) for span in group["spans"])
                           for group in groups) if groups else None)
            diagnostic.append({"requirement_id": requirement.get("requirement_id"),
                               "covered": covered,
                               "assessment_status": requirement.get("assessment_status")})
        approved = (approved_case and issue.get("assessment_status") == "approved" and bool(requirements)
                    and all(row.get("assessment_status") == "approved" for row in requirements))
        if approved:
            issues[issue["issue_id"]] = {"status": "assessed", "all_required_covered":
                all(row["covered"] is True for row in diagnostic), "requirements": diagnostic}
        else:
            issues[issue["issue_id"]] = _unassessed("draft_or_missing_requirements",
                all_required_covered=None, draft_diagnostics=diagnostic)
    if all(row["status"] == "assessed" for row in issues.values()):
        return {"status": "assessed", "all_required_covered":
                all(row["all_required_covered"] for row in issues.values()), "issues": issues}
    return _unassessed("draft_or_missing_issue_annotation", all_required_covered=None, issues=issues)


def evaluate_case(case: Mapping, observation: Mapping | None, chunks: Sequence[Mapping], k: int = 5) -> dict:
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if observation is not None and not isinstance(observation, Mapping):
        raise ValueError("observation must be a mapping or None")
    validation = validate_case_annotation(case, chunks)
    report = {"case_id": case.get("case_id") if isinstance(case, Mapping) else None,
              "annotation": validation,
              "annotation_status": case.get("assessment_status", "not_assessed") if isinstance(case, Mapping) else "not_assessed"}
    if validation["status"] != "valid" or observation is None:
        reason = "invalid_or_missing_annotation" if validation["status"] != "valid" else "missing_observation"
        return {**report, "retrieval": _unassessed(reason, k=k, hit=None, recall=None, mrr=None),
                "coverage": _unassessed(reason, all_required_covered=None, issues={}),
                "answer_quality": _unassessed(reason)}
    if case.get("assessment_status") == "approved":
        retrieval = retrieval_at_k(observation.get("retrieved_articles"), case.get("expected_articles"), k)
    else:
        retrieval = _unassessed("draft_expected_articles", k=k, hit=None, recall=None, mrr=None)
    judgments = observation.get("answer_judgments")
    if judgments is not None and not isinstance(judgments, Mapping):
        raise ValueError("answer judgments must be a mapping")
    quality = {}
    for name in ("false_supported", "wrong_document", "repealed_as_current", "hard_negative_used", "factual_preservation"):
        value = judgments.get(name) if isinstance(judgments, Mapping) else None
        if value is not None and type(value) is not bool:
            raise ValueError("answer judgments must be booleans")
        quality[name] = value
    if case.get("assessment_status") != "approved":
        answer = _unassessed("draft_annotation", **dict.fromkeys(quality))
    else:
        answer = {"status": "assessed" if any(value is not None for value in quality.values()) else "not_assessed",
                  "basis": "supplied_judgments_only", **quality}
    return {**report, "retrieval": retrieval,
            "coverage": evidence_coverage(case, observation.get("context_spans"), chunks),
            "answer_quality": answer}


def quantiles(values) -> dict:
    values = list(values)
    if not values:
        return _unassessed("missing_measurements", n=0, p50=None, p95=None)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("measurements must be finite nonnegative numbers")
    ordered = sorted(values)
    def percentile(fraction):
        offset = (len(ordered) - 1) * fraction
        low = math.floor(offset)
        high = math.ceil(offset)
        return ordered[low] + (ordered[high] - ordered[low]) * (offset - low)
    return {"status": "assessed", "n": len(values), "p50": percentile(.5), "p95": percentile(.95)}


def _rate(values, total: int) -> dict:
    present = [value for value in values if value is not None]
    if any(type(value) is not bool for value in present):
        raise ValueError("rate observations must be booleans")
    if not present:
        return _unassessed("missing_judgments", n=0, not_assessed=total, value=None)
    return {"status": "assessed", "n": len(present), "not_assessed": total - len(present),
            "value": sum(present) / len(present)}


def summarize_observations(observations: Sequence[Mapping | None]) -> dict:
    """Operational fields only; absence is not a zero count or success."""
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes, Mapping)):
        raise ValueError("observations must be a sequence of mappings")
    rows = [{} if row is None else row for row in observations]
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("observations must be mappings")
    total = len(rows)
    rates = {name: _rate([row.get(name) for row in rows], total)
             for name in ("fallback", "invalid_json", "budget_exhausted")}
    counters = {}
    for name in ("completion_calls", "rerank_pairs", "context_tokens"):
        values = [row[name] for row in rows if row.get(name) is not None]
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("counters must be nonnegative integers")
        counters[name] = {**quantiles(values), "total": sum(values) if values else None,
                          "not_assessed": total - len(values)}
    stage_maps = []
    for row in rows:
        stages_field = row.get("stage_seconds")
        if stages_field is None:
            stage_maps.append({})
        elif isinstance(stages_field, Mapping):
            stage_maps.append(stages_field)
        else:
            raise ValueError("stage_seconds must be a mapping")
    stages = sorted({stage for stage_map in stage_maps for stage in stage_map})
    stage_seconds = {}
    for stage in stages:
        values = [stage_map[stage] for stage_map in stage_maps if stage_map.get(stage) is not None]
        stage_seconds[stage] = {**quantiles(values), "not_assessed": total - len(values)}
    any_stage_assessed = any(info["status"] == "assessed" for info in stage_seconds.values())
    return {"operational_rates": rates, "counters": counters,
            "stage_seconds": {"status": "assessed" if any_stage_assessed else "not_assessed",
                              "stages": stage_seconds}}


def aggregate_case_reports(reports: Sequence[Mapping]) -> dict:
    assessed = [row["retrieval"] for row in reports if row["retrieval"]["status"] == "assessed"]
    ks = {row["retrieval"]["k"] for row in reports if "k" in row["retrieval"]}
    if len(ks) > 1:
        raise ValueError("retrieval comparisons require the same k")
    retrieval = ({"status": "assessed", "k": next(iter(ks)), "n": len(assessed),
                  "not_assessed": len(reports) - len(assessed),
                  **{name: sum(row[name] for row in assessed) / len(assessed)
                     for name in ("hit", "recall", "mrr")}}
                 if assessed else _unassessed("missing_approved_retrieval", n=0,
                     not_assessed=len(reports), k=next(iter(ks), None), hit=None, recall=None, mrr=None))
    coverage = _rate([row["coverage"].get("all_required_covered")
                      if row["coverage"]["status"] == "assessed" else None for row in reports], len(reports))
    answer = {name: _rate([row["answer_quality"].get(name)
                          if row["answer_quality"]["status"] == "assessed" else None for row in reports], len(reports))
              for name in ("false_supported", "wrong_document", "repealed_as_current", "hard_negative_used", "factual_preservation")}
    return {"retrieval": retrieval, "evidence_coverage": coverage, "answer_quality": answer,
            "annotation_statuses": sorted({row["annotation_status"] for row in reports}),
            "acceptance": "user_review_required"}


def evaluate_suite(artifact: Mapping, observations: Mapping[str, Mapping | None] | None, chunks: Sequence[Mapping], k: int = 5) -> dict:
    """Return content-free reports; questions/source quotes never enter output."""
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if observations is not None and not isinstance(observations, Mapping):
        raise ValueError("observations must be a mapping")
    obs_map = {} if observations is None else observations
    validation = validate_smoke_artifact(artifact, chunks)
    if validation["status"] != "valid":
        return _unassessed("invalid_or_missing_artifact", annotation=validation, cases=[],
                           acceptance="user_review_required")
    reports = [evaluate_case(case, obs_map.get(case["case_id"]), chunks, k) for case in artifact["cases"]]
    return {"status": "reported", "annotation": validation, "cases": reports,
            **aggregate_case_reports(reports),
            **summarize_observations([obs_map.get(case["case_id"], {}) for case in artifact["cases"]])}
