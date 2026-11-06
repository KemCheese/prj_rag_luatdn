"""Zero-retry structured completion, separate from the existing reader helper."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import TypeVar

import httpx
from openai import APIError
from pydantic import BaseModel, ValidationError

from ragdn import llm
from ragdn.query_models import RequestBudget, QueryServiceError, StructuredCompletionError

T = TypeVar("T", bound=BaseModel)


def structured_completion(
    system: str,
    user: str,
    *,
    schema: type[T],
    budget: RequestBudget,
    client=None,
    model: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> T:
    """Count each sent request, reject invalid output, never repair or log bodies.

    The orchestrator must pack the complete prompt against confirmed context
    before calling this helper. Transport timeouts are not wall-clock deadlines.
    """
    config = budget.config
    budget.check_stage(now=clock())
    base_client = llm._client if client is None else client
    request_client = base_client.with_options(
        max_retries=0,
        timeout=httpx.Timeout(timeout=config.call_timeout, connect=config.connect_timeout),
    )
    kwargs = {
        "model": model or os.environ.get("LLM_MODEL", "qwen2.5:3b"),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": config.output_tokens,
        "temperature": llm.LLM_TEMPERATURE,
        "top_p": llm.LLM_TOP_P,
        "response_format": {"type": "json_object"},
    }
    if llm.LLM_SEED:
        kwargs["seed"] = int(llm.LLM_SEED)
    budget.begin_completion(now=clock())
    started = clock()
    try:
        response = request_client.chat.completions.create(**kwargs)
    except (APIError, httpx.TransportError, TimeoutError):
        raise QueryServiceError("structured_service_error") from None
    finally:
        budget.stage_seconds["structured_completion"] = (
            budget.stage_seconds.get("structured_completion", 0.0) + max(0.0, clock() - started)
        )
    if not response.choices:
        raise StructuredCompletionError("empty_response")
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise StructuredCompletionError("truncated_response")
    if choice.finish_reason != "stop":
        raise StructuredCompletionError("incomplete_response")
    content = choice.message.content
    if not isinstance(content, str) or not content.strip():
        raise StructuredCompletionError("empty_response")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        raise StructuredCompletionError("invalid_json") from None
    try:
        return schema.model_validate(payload)
    except ValidationError:
        raise StructuredCompletionError("invalid_schema") from None
