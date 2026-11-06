"""Validated configuration for the bounded query-analysis/retrieval stages."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QueryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    mode: Literal["adaptive", "legacy"] = "adaptive"
    max_issues: int = Field(default=6, gt=0, le=6)
    max_initial_queries: int = Field(default=7, gt=0)
    max_supplementary_queries: int = Field(default=6, ge=0)
    candidates_per_channel: int = Field(default=20, gt=0)
    parents_per_query: int = Field(default=12, gt=0)
    max_active_parents: int = Field(default=48, gt=0)
    max_rerank_pairs: int = Field(default=96, gt=0)
    initial_rerank_pairs: int = Field(default=72, gt=0)
    supplementary_rerank_pairs: int = Field(default=24, ge=0)
    max_evidence_groups: int = Field(default=12, gt=0)
    max_completions: int = Field(default=3, gt=0, le=3)
    max_supplementary_rounds: int = Field(default=1, ge=0, le=1)
    query_tokens: int = Field(default=96, gt=0)
    passage_tokens: int = Field(default=384, gt=0)
    overlap_tokens: int = Field(default=64, ge=0)
    pair_tokens: int = Field(default=512, gt=0, le=512)
    call_timeout: float = Field(default=200.0, gt=0)
    connect_timeout: float = Field(default=10.0, gt=0)
    stage_timeout: float = Field(default=600.0, gt=0)
    output_tokens: int = Field(default=2000, gt=0)
    template_reserve_tokens: int = Field(default=512, ge=512)
    context_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def coherent_limits(self) -> Self:
        if self.initial_rerank_pairs + self.supplementary_rerank_pairs > self.max_rerank_pairs:
            raise ValueError("initial plus supplementary rerank pairs exceeds total")
        if self.overlap_tokens >= self.passage_tokens:
            raise ValueError("overlap_tokens must be smaller than passage_tokens")
        if self.query_tokens + self.passage_tokens >= self.pair_tokens:
            raise ValueError("pair_tokens must leave room for special tokens")
        if self.parents_per_query > self.max_active_parents:
            raise ValueError("parents_per_query exceeds max_active_parents")
        if self.connect_timeout > self.call_timeout:
            raise ValueError("connect_timeout exceeds call_timeout")
        return self

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Self:
        """Read only named RAG_QUERY_* values; never alter environment/config files."""
        source = os.environ if environ is None else environ
        values = {
            name: source[f"RAG_QUERY_{name.upper()}"]
            for name in cls.model_fields
            if f"RAG_QUERY_{name.upper()}" in source
        }
        return cls.model_validate(values)
