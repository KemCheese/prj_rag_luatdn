"""Request-local query contracts; the reference schema belongs exclusively to P1."""
from __future__ import annotations

import time
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ragdn.article_references import ArticleReference
from ragdn.query_config import QueryConfig


class QueryPipelineError(RuntimeError):
    """Expected new-stage failure with a content-free fallback reason."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class QueryBudgetExceeded(QueryPipelineError):
    pass


class StructuredCompletionError(QueryPipelineError):
    pass


class QueryServiceError(QueryPipelineError):
    pass


class QueryAnalysisError(QueryPipelineError):
    pass


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class IssueStatus(StrEnum):
    SUPPORTED = "supported"
    CONDITIONAL = "conditional"
    INSUFFICIENT = "insufficient"
    AMBIGUOUS = "ambiguous"
    UNCERTAIN = "uncertain"
    NOT_PROCESSED = "not_processed"
    NOT_ASSESSED = "not_assessed"


class ProcessingStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    NOT_ASSESSED = "not_assessed"


class ProcessingMode(StrEnum):
    DIRECT = "direct"
    ENHANCED = "enhanced"
    LEGACY = "legacy"
    BASELINE_FALLBACK = "baseline_fallback"


class PartialReason(StrEnum):
    GENERATION_PRUNED = "generation_pruned"
    EVIDENCE_MISSING = "evidence_missing"
    INFORMATION_MISSING = "information_missing"
    AMBIGUOUS_SCOPE = "ambiguous_scope"
    BUDGET_EXHAUSTED = "budget_exhausted"


class FactProposal(Contract):
    fact_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    subject: str | None = None
    relation: str | None = None
    is_unknown: bool = False


class FactRecord(FactProposal):
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_span(self) -> Self:
        if self.end <= self.start or self.end - self.start != len(self.quote):
            raise ValueError("fact span must match quote length")
        return self


class IssueScope(Contract):
    subjects: list[str] = Field(default_factory=list)
    company_types: list[str] = Field(default_factory=list)
    doc_ids: list[str] = Field(default_factory=list)
    time_quote: str | None = None


class IssueRecord(Contract):
    issue_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    fact_ids: list[str] = Field(default_factory=list)
    scope: IssueScope = Field(default_factory=IssueScope)
    main_query: str = Field(min_length=1)
    backup_query: str | None = Field(default=None, min_length=1)


class PlannerIssue(IssueRecord):
    backup_query: str = Field(min_length=1)


class PlannerOutput(Contract):
    facts: list[FactProposal]
    issues: list[PlannerIssue] = Field(min_length=1)
    reference_quotes: list[str] = Field(default_factory=list)
    unprocessed_parts: list[str] = Field(default_factory=list)


class NotProcessedIssue(Contract):
    issue_id: str
    title: str
    reason: str = "budget_exhausted"
    status: Literal["not_processed"] = "not_processed"


class QuerySpec(Contract):
    query_id: str
    text: str
    issue_ids: list[str]
    scope: IssueScope
    kind: Literal["main", "backup"]


class QuestionPlan(Contract):
    original_question: str = Field(min_length=1)
    facts: list[FactRecord]
    issues: list[IssueRecord]
    references: list[ArticleReference] = Field(default_factory=list)
    queries: list[QuerySpec] = Field(default_factory=list)
    not_processed: list[NotProcessedIssue] = Field(default_factory=list)
    unprocessed_parts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SourceSpan(Contract):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_span(self) -> Self:
        if self.end <= self.start or self.end - self.start != len(self.quote):
            raise ValueError("source span must match quote length")
        return self


class EvidenceRecord(Contract):
    evidence_id: str
    issue_id: str
    parent_chunk_id: str
    doc_id: str
    dieu: str
    source_spans: list[SourceSpan] = Field(min_length=1)
    khoan: str | None = None
    diem: str | None = None
    window_ids: list[str] = Field(default_factory=list)
    unit_ids: list[str] = Field(default_factory=list)
    condition_labels: list[str] = Field(default_factory=list)
    purpose: str = "current_law"


class MissingInformation(Contract):
    description: str
    source_evidence_ids: list[str] = Field(default_factory=list)
    scope_unresolved: bool = False
    source_grounded: bool = False


class IssueAssessment(Contract):
    issue_id: str
    status: IssueStatus
    evidence_ids: list[str] = Field(default_factory=list)
    missing_information: list[MissingInformation] = Field(default_factory=list)
    missing_scope: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class BudgetSnapshot(Contract):
    completion_calls: int = 0
    initial_queries: int = 0
    supplementary_queries: int = 0
    initial_rerank_pairs: int = 0
    supplementary_rerank_pairs: int = 0
    supplementary_rounds: int = 0
    elapsed_seconds: float = 0.0
    not_processed: list[str] = Field(default_factory=list)


class RequestBudget(BaseModel):
    """One instance per request; reserve immediately before each bounded action."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)
    config: QueryConfig = Field(default_factory=QueryConfig)
    started_at: float = Field(default_factory=time.monotonic)
    completion_calls: int = Field(default=0, ge=0)
    initial_queries: int = Field(default=0, ge=0)
    supplementary_queries: int = Field(default=0, ge=0)
    initial_rerank_pairs: int = Field(default=0, ge=0)
    supplementary_rerank_pairs: int = Field(default=0, ge=0)
    supplementary_rounds: int = Field(default=0, ge=0)
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    not_processed: list[str] = Field(default_factory=list)

    def check_stage(self, *, now: float | None = None) -> None:
        instant = time.monotonic() if now is None else now
        if instant - self.started_at >= self.config.stage_timeout:
            raise QueryBudgetExceeded("stage_deadline_exhausted")

    def begin_completion(self, *, now: float | None = None) -> None:
        self.check_stage(now=now)
        if self.completion_calls >= self.config.max_completions:
            raise QueryBudgetExceeded("completion_budget_exhausted")
        self.completion_calls += 1

    def record_query(self, *, supplementary: bool = False, now: float | None = None) -> None:
        self.check_stage(now=now)
        name = "supplementary_queries" if supplementary else "initial_queries"
        limit = self.config.max_supplementary_queries if supplementary else self.config.max_initial_queries
        if getattr(self, name) >= limit:
            raise QueryBudgetExceeded("query_budget_exhausted")
        setattr(self, name, getattr(self, name) + 1)

    def record_rerank(self, pairs: int, *, supplementary: bool = False, now: float | None = None) -> None:
        self.check_stage(now=now)
        if pairs < 0:
            raise ValueError("pairs must be nonnegative")
        name = "supplementary_rerank_pairs" if supplementary else "initial_rerank_pairs"
        limit = self.config.supplementary_rerank_pairs if supplementary else self.config.initial_rerank_pairs
        if (getattr(self, name) + pairs > limit or
                self.initial_rerank_pairs + self.supplementary_rerank_pairs + pairs > self.config.max_rerank_pairs):
            raise QueryBudgetExceeded("rerank_budget_exhausted")
        setattr(self, name, getattr(self, name) + pairs)

    def begin_supplementary(self, *, now: float | None = None) -> None:
        self.check_stage(now=now)
        if self.supplementary_rounds >= self.config.max_supplementary_rounds:
            raise QueryBudgetExceeded("supplementary_budget_exhausted")
        self.supplementary_rounds += 1

    def snapshot(self, *, now: float | None = None) -> BudgetSnapshot:
        instant = time.monotonic() if now is None else now
        names = ("completion_calls", "initial_queries", "supplementary_queries", "initial_rerank_pairs",
                 "supplementary_rerank_pairs", "supplementary_rounds", "not_processed")
        return BudgetSnapshot(**{name: getattr(self, name) for name in names},
                              elapsed_seconds=max(0.0, instant - self.started_at))


class ProcessingReport(Contract):
    mode: ProcessingMode
    route_reason: str
    status: ProcessingStatus = ProcessingStatus.NOT_ASSESSED
    issues: list[IssueAssessment] = Field(default_factory=list)
    missing_information: list[MissingInformation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    partial_reasons: list[PartialReason] = Field(default_factory=list)
    budget: BudgetSnapshot = Field(default_factory=BudgetSnapshot)
    trace_id: str | None = None


class RouteDecision(Contract):
    mode: Literal["direct", "enhanced"]
    route_reason: str
    references: list[ArticleReference] = Field(default_factory=list)
