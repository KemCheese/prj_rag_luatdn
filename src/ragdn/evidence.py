"""Code-owned source locators and conservative validation of mapper proposals.

Structural provenance checks constrain a proposal; they do not prove that its
legal interpretation is correct. No model, corpus mutation or retrieval occurs.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
import re
import unicodedata
from typing import Literal

from pydantic import Field

from ragdn.loai_hinh import is_conflict
from ragdn.query_models import (
    Contract, EvidenceRecord, IssueAssessment, IssueRecord, IssueStatus,
    MissingInformation, PartialReason, ProcessingStatus, QuestionPlan, SourceSpan,
)
from ragdn.source_windows import SourceWindow, WindowGroup

__all__ = [
    "Coverage",
    "EvidenceCatalog",
    "EvidenceProposal",
    "IssueMapping",
    "MapperOutput",
    "MissingProposal",
    "Rejection",
    "SuppliedGroup",
    "article",
    "coverage_status",
    "document_metadata",
    "fold",
    "group_error",
    "merge_coverage",
    "validate_mapping",
]


def fold(text: str) -> str:
    return re.sub(r"\s+", " ", "".join(
        char for char in unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
        if unicodedata.category(char) != "Mn"
    )).strip()


def article(text: str) -> str:
    return unicodedata.normalize("NFC", text.casefold()).replace("dd", "đ").strip()


class EvidenceProposal(Contract):
    evidence_id: str = Field(min_length=1)
    issue_id: str
    group_id: str
    unit_ids: list[str] = Field(min_length=1)
    window_ids: list[str] = Field(default_factory=list)
    doc_id: str
    dieu: str
    khoan: str | None = None
    diem: str | None = None
    purpose: Literal["current_law", "history", "repeal"] = "current_law"
    condition_quotes: list[str] = Field(default_factory=list)


class MissingProposal(Contract):
    description: str = Field(min_length=1)
    source_evidence_ids: list[str] = Field(default_factory=list)
    condition_quote: str | None = None
    scope_field: Literal["subjects", "company_types", "doc_ids", "time_quote"] | None = None
    scope_fact_id: str | None = None
    scope_unresolved: bool = False
    source_grounded: bool = False


class IssueMapping(Contract):
    issue_id: str
    status: Literal["supported", "conditional", "insufficient", "ambiguous", "uncertain"]
    evidence_ids: list[str] = Field(default_factory=list)
    missing_information: list[MissingProposal] = Field(default_factory=list)


class MapperOutput(Contract):
    evidence: list[EvidenceProposal] = Field(default_factory=list)
    issues: list[IssueMapping]


@dataclass(frozen=True)
class SuppliedGroup:
    group: WindowGroup
    windows: tuple[SourceWindow, ...]
    issue_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceCatalog:
    parents: dict[str, dict]
    groups: dict[str, SuppliedGroup]
    documents: dict[str, dict]


@dataclass(frozen=True)
class Rejection:
    proposal_id: str
    issue_id: str
    reason: str


@dataclass(frozen=True)
class Coverage:
    evidence: tuple[EvidenceRecord, ...]
    assessments: tuple[IssueAssessment, ...]
    rejections: tuple[Rejection, ...] = ()


def document_metadata(manifest) -> dict[str, dict]:
    if manifest is None:
        return {}
    value = manifest.model_dump() if hasattr(manifest, "model_dump") else manifest
    documents = value.get("documents", [value] if "doc_id" in value else []) if isinstance(value, Mapping) else value
    return {item["doc_id"]: item for document in documents
            for item in [document.model_dump() if hasattr(document, "model_dump") else dict(document)]}


def group_error(supplied: SuppliedGroup, parents: dict[str, dict]) -> str | None:
    group = supplied.group
    parent = parents.get(group.parent_chunk_id)
    if parent is None or not isinstance(parent.get("text"), str) or not parent["text"]:
        return "source_parent_missing"
    text = parent["text"]
    if not group.atomic or not group.units:
        return "source_group_not_atomic"
    if len({unit.unit_id for unit in group.units}) != len(group.units):
        return "source_unit_duplicate"
    previous_end = 0
    for unit in group.units:
        if (type(unit.start) is not int or type(unit.end) is not int
                or not previous_end <= unit.start < unit.end <= len(text)
                or text[unit.start:unit.end] != unit.text):
            return "source_unit_span_invalid"
        previous_end = unit.end
    windows = {window.window_id: window for window in supplied.windows}
    if not group.window_ids or len(windows) != len(supplied.windows) or not set(group.window_ids) <= windows.keys():
        return "source_window_missing"
    for window_id in group.window_ids:
        window = windows[window_id]
        if (window.parent_chunk_id != group.parent_chunk_id
                or type(window.start) is not int or type(window.end) is not int
                or not 0 <= window.start < window.end <= len(text)
                or text[window.start:window.end] != window.text):
            return "source_window_span_invalid"
    clause_starts = [match.group(1).lower().replace("dd", "đ")
                     for match in re.finditer(r"(?m)^\s*(\d+(?:dd|[a-zđ])?)\.\s+", text)]
    if len(set(clause_starts)) != len(clause_starts):
        contiguous = all(left.end == right.start for left, right in zip(group.units, group.units[1:]))
        if (group.units[0].start != 0 or group.units[-1].end != len(text)
                or not contiguous or set(group.window_ids) != windows.keys()):
            return "source_group_not_atomic"
    return None


_HEADER_OR_REFS = re.compile(
    r"^(?:(?:phan|chuong|muc|tieu muc)\s+(?:[ivxlcdm]+|\d+)\b|"
    r"dieu\s+\d+(?:dd|[a-z])?\b|"
    r"(?:tham chieu|tham khao|xem|xem them|refs|chi dan tham chieu)\b)"
)


def _substantive(text: str) -> bool:
    lines = []
    for raw_line in text.splitlines():
        stripped = re.sub(r"^\s*(?:\d+(?:dd|[a-zđ])?\.\s*|(?:dd|[a-zđ])\)\s*|[\[\(\-\*•]+\s*)*", "", raw_line)
        if stripped and not _HEADER_OR_REFS.match(fold(stripped)):
            lines.append(stripped)
    body = " ".join(lines)
    return len(re.findall(r"[^\W\d_]+", body)) >= 4 and len(body.strip()) >= 18


def _requested_history(issue: IssueRecord, plan: QuestionPlan) -> bool:
    other_facts = {fid for other in plan.issues if other.issue_id != issue.issue_id for fid in other.fact_ids}
    facts = " ".join(fact.quote for fact in plan.facts if fact.fact_id in issue.fact_ids and fact.fact_id not in other_facts)
    return bool(re.search(r"\b(?:bai bo|lich su|het hieu luc|truoc khi|thay the)\b",
                          fold(f"{issue.title} {issue.main_query} {facts}")))


def _effective_after_scope(effective: str | None, time_quote: str | None) -> bool:
    if not effective or not time_quote:
        return False
    try:
        starts = date.fromisoformat(effective)
    except ValueError:
        return True
    folded = fold(time_quote)
    parsed_dates: list[date] = []
    for year, month, day in re.findall(r"(?<!\d)(\d{4})[/-](\d{1,2})[/-](\d{1,2})(?!\d)", time_quote):
        try:
            parsed_dates.append(date(int(year), int(month), int(day)))
        except ValueError:
            return True
    for day, month, year in [
        *re.findall(r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?!\d)", time_quote),
        *re.findall(r"\bngay\s+(\d{1,2})\s+thang\s+(\d{1,2})\s+nam\s+(\d{4})\b", folded),
    ]:
        try:
            parsed_dates.append(date(int(year), int(month), int(day)))
        except ValueError:
            return True
    if parsed_dates:
        return all(starts > target for target in parsed_dates)
    months = [(int(year), int(month)) for month, year in [
        *re.findall(r"\bthang\s+(\d{1,2})\s+(?:nam\s+)?(\d{4})\b", folded),
        *re.findall(r"(?<!\d)(\d{1,2})/(\d{4})(?!\d)", time_quote),
    ] if 1 <= int(month) <= 12]
    if months:
        return all((starts.year, starts.month) > ym for ym in months)
    years = re.findall(r"(?<!\d)(\d{4})(?!\d)", time_quote)
    return bool(years) and all(starts.year > int(year) for year in years)


def _evidence_error(proposal: EvidenceProposal, issue: IssueRecord, plan: QuestionPlan,
                    catalog: EvidenceCatalog) -> str | None:
    supplied = catalog.groups.get(proposal.group_id)
    if supplied is None:
        return "evidence_group_not_supplied"
    if issue.issue_id not in supplied.issue_ids:
        return "evidence_issue_not_supplied"
    reason = group_error(supplied, catalog.parents)
    if reason:
        return reason
    group = supplied.group
    if len(set(proposal.unit_ids)) != len(proposal.unit_ids) or set(proposal.unit_ids) != {unit.unit_id for unit in group.units}:
        return "evidence_atomic_group_incomplete"
    if len(set(proposal.window_ids)) != len(proposal.window_ids) or not set(proposal.window_ids) <= set(group.window_ids):
        return "evidence_window_not_supplied"
    if proposal.window_ids and set(proposal.window_ids) != set(group.window_ids):
        return "evidence_atomic_group_incomplete"
    parent = catalog.parents[group.parent_chunk_id]
    if proposal.doc_id != parent["doc_id"] or article(proposal.dieu) != article(str(parent["dieu"])):
        return "evidence_document_article_mismatch"
    meta = catalog.documents.get(parent["doc_id"], {})
    for key in ("vai_tro", "sua_doi_cho", "ngay_hieu_luc"):
        if parent.get(key) is not None and meta.get(key) is not None and parent[key] != meta[key]:
            return "evidence_document_metadata_mismatch"
    role = meta.get("vai_tro", parent.get("vai_tro"))
    target = meta.get("sua_doi_cho", parent.get("sua_doi_cho"))
    if role not in {"chinh", "sua_doi"}:
        return "evidence_document_role_invalid"
    if role == "chinh" and target:
        return "evidence_amendment_target_unknown"
    if role == "sua_doi":
        if not target or target == parent["doc_id"]:
            return "evidence_amendment_target_unknown"
        if target in catalog.documents and catalog.documents[target].get("vai_tro", "chinh") != "chinh":
            return "evidence_amendment_target_unknown"
    if issue.scope.doc_ids and parent["doc_id"] not in issue.scope.doc_ids:
        if role != "sua_doi" or target not in issue.scope.doc_ids:
            return "evidence_document_scope_conflict"
    if issue.scope.company_types and all(is_conflict(parent.get("loai_hinh"), company_type) for company_type in issue.scope.company_types):
        return "evidence_company_scope_conflict"
    if proposal.purpose == "current_law":
        status = meta.get("trang_thai", parent.get("trang_thai", "con_hieu_luc"))
        if parent.get("hieu_luc") is not True or status != "con_hieu_luc":
            return "evidence_not_current"
        if _effective_after_scope(meta.get("ngay_hieu_luc", parent.get("ngay_hieu_luc")), issue.scope.time_quote):
            return "evidence_not_yet_effective"
    elif not _requested_history(issue, plan):
        return "evidence_history_not_requested"
    text = "\n".join(unit.text for unit in group.units)
    if not any(_substantive(unit.text) for unit in group.units):
        return "evidence_not_substantive"
    clause_text = text
    if proposal.khoan is not None:
        matches = [unit for unit in group.units if unit.khoan and article(unit.khoan) == article(proposal.khoan)]
        if len(matches) != 1 or not re.match(r"^\s*" + re.escape(article(proposal.khoan)) + r"\.\s", unicodedata.normalize("NFC", matches[0].text.casefold()).replace("dd", "đ")):
            return "evidence_clause_not_recognizable"
        if not _substantive(matches[0].text):
            return "evidence_not_substantive"
        clause_text = matches[0].text
    if proposal.diem is not None:
        point_matches = list(re.finditer(r"(?m)^\s*" + re.escape(article(proposal.diem)) + r"\)\s",
                                         unicodedata.normalize("NFC", clause_text.casefold()).replace("dd", "đ")))
        if len(point_matches) != 1:
            return "evidence_point_not_recognizable"
    if any(not quote.strip() or not any(quote in unit.text for unit in group.units) for quote in proposal.condition_quotes):
        return "evidence_condition_not_present"
    return None


_CONDITION = re.compile(r"\b(?:neu|khi|tru|ngoai tru|truong hop|dieu kien|dieu le|it nhat|toi thieu|toi da|ty le|thoi han|phu thuoc)\b")
_SCOPE_MARKERS = {"subjects": r"chu the|cong ty|doanh nghiep", "company_types": r"loai hinh",
                  "doc_ids": r"van ban|luat|nghi dinh", "time_quote": r"thoi diem|thoi gian|nam|ngay"}


def _missing(proposal: MissingProposal, issue: IssueRecord, plan: QuestionPlan,
             evidence: dict[str, EvidenceRecord], id_prefix: str) -> MissingInformation | None:
    linked = [evidence[identifier] for identifier in proposal.source_evidence_ids
              if identifier in evidence and evidence[identifier].issue_id == issue.issue_id]
    if (proposal.source_evidence_ids and len(linked) == len(proposal.source_evidence_ids) == len(set(proposal.source_evidence_ids))
            and proposal.condition_quote and _CONDITION.search(fold(proposal.condition_quote))
            and all(any(proposal.condition_quote in span.quote for span in record.source_spans) for record in linked)):
        return MissingInformation(description=proposal.description,
                                  source_evidence_ids=[id_prefix + identifier for identifier in proposal.source_evidence_ids],
                                  source_grounded=True)
    field = proposal.scope_field
    if field and not getattr(issue.scope, field):
        if proposal.scope_fact_id is not None:
            candidates = [fact for fact in plan.facts if fact.fact_id == proposal.scope_fact_id and fact.fact_id in issue.fact_ids]
        else:
            candidates = [fact for fact in plan.facts if fact.fact_id in issue.fact_ids]
        unresolved_reference = field == "doc_ids" and proposal.scope_fact_id is None and any(reference.ambiguous for reference in plan.references)
        if unresolved_reference or any(fact.is_unknown and re.search(_SCOPE_MARKERS[field], fold(fact.quote)) for fact in candidates):
            return MissingInformation(description=proposal.description, scope_unresolved=True)
    return None


def validate_mapping(plan: QuestionPlan, proposal: MapperOutput, catalog: EvidenceCatalog, *,
                     unseen_issue_ids: set[str] | None = None, id_prefix: str = "") -> Coverage:
    unseen = unseen_issue_ids or set()
    issues = {issue.issue_id: issue for issue in plan.issues}
    records: dict[str, EvidenceRecord] = {}
    rejections: list[Rejection] = []
    duplicate_ids = {key for key, count in Counter(item.evidence_id for item in proposal.evidence).items() if count > 1}
    for item in proposal.evidence:
        issue = issues.get(item.issue_id)
        reason = ("evidence_duplicate_id" if item.evidence_id in duplicate_ids else
                  "evidence_unknown_issue" if issue is None else _evidence_error(item, issue, plan, catalog))
        if reason:
            rejections.append(Rejection(item.evidence_id, item.issue_id, reason))
            continue
        group = catalog.groups[item.group_id].group
        records[item.evidence_id] = EvidenceRecord(
            evidence_id=id_prefix + item.evidence_id, issue_id=item.issue_id,
            parent_chunk_id=group.parent_chunk_id, doc_id=item.doc_id, dieu=article(item.dieu),
            khoan=article(item.khoan) if item.khoan is not None else None,
            diem=article(item.diem) if item.diem is not None else None,
            source_spans=[SourceSpan(start=unit.start, end=unit.end, quote=unit.text) for unit in group.units],
            window_ids=list(group.window_ids), unit_ids=[unit.unit_id for unit in group.units],
            condition_labels=item.condition_quotes, purpose=item.purpose,
        )
    mappings: dict[str, list[IssueMapping]] = {}
    for item in proposal.issues:
        mappings.setdefault(item.issue_id, []).append(item)
        if item.issue_id not in issues:
            rejections.append(Rejection(item.issue_id, item.issue_id, "mapping_unknown_issue"))
    assessments = []
    for issue in plan.issues:
        entries = mappings.get(issue.issue_id, [])
        if len(entries) != 1:
            assessments.append(IssueAssessment(issue_id=issue.issue_id, status="uncertain", reasons=["mapping_issue_missing_or_duplicate"]))
            continue
        item = entries[0]
        selected = [identifier for identifier in dict.fromkeys(item.evidence_ids)
                    if identifier in records and records[identifier].issue_id == issue.issue_id]
        reasons = []
        if len(selected) != len(set(item.evidence_ids)) or any(r.issue_id == issue.issue_id for r in rejections):
            reasons.append("mapping_evidence_rejected")
        missing = []
        for proposed_missing in item.missing_information:
            verified = _missing(proposed_missing, issue, plan, records, id_prefix)
            if verified:
                missing.append(verified)
            else:
                reasons.append("missing_information_not_grounded")
        status = IssueStatus(item.status)
        if status in {IssueStatus.SUPPORTED, IssueStatus.CONDITIONAL} and not selected:
            status = IssueStatus.INSUFFICIENT
            reasons.append("evidence_missing")
        elif status == IssueStatus.SUPPORTED:
            if any(info.scope_unresolved for info in missing):
                status = IssueStatus.AMBIGUOUS
            elif any(info.source_grounded for info in missing):
                status = IssueStatus.CONDITIONAL
        elif status == IssueStatus.CONDITIONAL:
            if any(info.scope_unresolved for info in missing):
                status = IssueStatus.AMBIGUOUS
            elif not any(info.source_grounded for info in missing):
                status = IssueStatus.UNCERTAIN
                reasons.append("conditional_information_not_grounded")
        elif status == IssueStatus.AMBIGUOUS:
            if not any(info.scope_unresolved for info in missing):
                status = IssueStatus.UNCERTAIN
                reasons.append("ambiguous_scope_not_grounded")
        if reasons and status in {IssueStatus.SUPPORTED, IssueStatus.CONDITIONAL}:
            status = IssueStatus.UNCERTAIN if selected else IssueStatus.INSUFFICIENT
        if issue.issue_id in unseen:
            reasons.append("source_spans_unseen")
            if status == IssueStatus.SUPPORTED:
                status = IssueStatus.UNCERTAIN
        assessments.append(IssueAssessment(issue_id=issue.issue_id, status=status,
                           evidence_ids=[id_prefix + identifier for identifier in selected],
                           missing_information=missing,
                           missing_scope=[info.description for info in missing if info.scope_unresolved],
                           reasons=list(dict.fromkeys(reasons))))
    assessments.extend(IssueAssessment(issue_id=item.issue_id, status="not_processed", reasons=[item.reason])
                       for item in plan.not_processed)
    return Coverage(tuple(records.values()), tuple(assessments), tuple(rejections))


def merge_coverage(initial: Coverage, rechecked: Coverage, eligible_issue_ids: set[str]) -> Coverage:
    initial_by_id = {item.issue_id: item for item in initial.assessments}
    effective_eligible = {
        issue_id for issue_id in eligible_issue_ids
        if issue_id in initial_by_id
        and initial_by_id[issue_id].status in {IssueStatus.INSUFFICIENT, IssueStatus.UNCERTAIN}
    }
    assessments = dict(initial_by_id)
    for item in rechecked.assessments:
        if item.issue_id not in effective_eligible:
            continue
        prior = initial_by_id[item.issue_id]
        if prior.evidence_ids and item.status in {IssueStatus.SUPPORTED, IssueStatus.CONDITIONAL, IssueStatus.UNCERTAIN}:
            merged_ids = list(dict.fromkeys([*prior.evidence_ids, *item.evidence_ids]))
            item = item.model_copy(update={"evidence_ids": merged_ids})
        assessments[item.issue_id] = item
    records = {record.evidence_id: record for record in initial.evidence}
    records.update((record.evidence_id, record) for record in rechecked.evidence if record.issue_id in effective_eligible)
    return Coverage(tuple(records.values()), tuple(assessments.values()), (*initial.rejections, *rechecked.rejections))


def coverage_status(coverage: Coverage) -> tuple[ProcessingStatus, list[PartialReason]]:
    statuses = [assessment.status for assessment in coverage.assessments]
    if statuses and all(status == IssueStatus.NOT_ASSESSED for status in statuses):
        return ProcessingStatus.NOT_ASSESSED, []
    answered = any(status in {IssueStatus.SUPPORTED, IssueStatus.CONDITIONAL} for status in statuses)
    if statuses and all(status == IssueStatus.SUPPORTED for status in statuses):
        return ProcessingStatus.SUPPORTED, []
    reasons = []
    if any(status in {IssueStatus.INSUFFICIENT, IssueStatus.UNCERTAIN} for status in statuses):
        reasons.append(PartialReason.EVIDENCE_MISSING)
    if any(status == IssueStatus.CONDITIONAL for status in statuses):
        reasons.append(PartialReason.INFORMATION_MISSING)
    if any(status == IssueStatus.AMBIGUOUS for status in statuses):
        reasons.append(PartialReason.AMBIGUOUS_SCOPE)
    if any(status == IssueStatus.NOT_PROCESSED or "source_spans_unseen" in assessment.reasons
           for status, assessment in zip(statuses, coverage.assessments)):
        reasons.append(PartialReason.BUDGET_EXHAUSTED)
    return (ProcessingStatus.PARTIAL if answered else ProcessingStatus.INSUFFICIENT), reasons
