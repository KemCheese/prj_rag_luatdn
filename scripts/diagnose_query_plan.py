"""User-run, single-completion planner diagnosis; no retrieval or reader calls.

Copy this file alone into another checkout's scripts directory and run it with
that checkout's Python environment. Import and --help perform no model request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive confirmed context size")
    return number


def _load_environment() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)


def _metadata(question: str) -> dict:
    return {"question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "question_length": len(question), "model": None,
            "context_tokens": None, "context_method": None,
            "new_completion_calls": 0, "quote_checks": [],
            "validation_reason": None, "error_reason": None,
            "parsed_plan_status": "not_captured", "schema_errors": []}


class _CapturedClient:
    """Forward the helper's request unchanged and retain only response content."""

    def __init__(self, client, capture):
        self._client, self._capture = client, capture
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def __getattr__(self, name):
        return getattr(self._client, name)

    def with_options(self, **kwargs):
        return _CapturedClient(self._client.with_options(**kwargs), self._capture)

    def _create(self, *args, **kwargs):
        response = self._client.chat.completions.create(*args, **kwargs)
        self._capture["available"] = True
        self._capture["content"] = response.choices[0].message.content if response.choices else None
        return response


def _inspect_response(capture: dict, schema) -> tuple[object, str, list[dict]]:
    """Parse the same response without repair; error output contains no inputs."""
    from pydantic import ValidationError

    content = capture.get("content")
    if not isinstance(content, str) or not content.strip():
        return None, "response_unavailable", []

    def invalid_constant(_):
        raise ValueError("nonstandard_json_constant")

    try:
        parsed = json.loads(content, parse_constant=invalid_constant)
    except (json.JSONDecodeError, ValueError):
        return None, "invalid_json", []

    def property_names(value):
        if isinstance(value, dict):
            yield from value.get("properties", {})
            for nested in value.values():
                yield from property_names(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from property_names(nested)

    allowed = set(property_names(schema.model_json_schema()))
    try:
        schema.model_validate(parsed)
    except ValidationError as error:
        errors = [{"loc": [part if isinstance(part, int) or part in allowed else "<unknown_field>"
                           for part in item["loc"]], "type": item["type"]}
                  for item in error.errors(include_url=False, include_context=False, include_input=False)]
        return parsed, "schema_invalid", errors
    return parsed, "schema_valid", []


def diagnose(question: str, *, context_tokens: int | None = None, show_quotes: bool = False, show_plan: bool = False,
             completion=None, validator=None, resolver=None, client=None) -> tuple[dict, int]:
    """Capture one typed proposal, then validate that identical object locally.

    Return0 for a valid plan,1 for expected preflight/completion/validation
    failure,2 for invalid configuration. Unexpected programming errors propagate.
    Optional callables/client support offline tests without altering runtime code.
    """
    from pydantic import ValidationError
    from ragdn.article_references import resolve_article_references
    from ragdn.query_analysis import build_planner_prompt, validate_plan
    from ragdn.query_config import QueryConfig
    from ragdn.query_llm import llm, structured_completion
    from ragdn.query_models import PlannerOutput, QueryPipelineError, RequestBudget
    from ragdn.query_pipeline import ReaderContext

    report = _metadata(question)
    report["model"] = os.environ.get("LLM_MODEL", "qwen2.5:3b")
    try:
        config = QueryConfig.from_env()
        if context_tokens is not None:
            config = QueryConfig.model_validate({**config.model_dump(), "context_tokens": context_tokens})
        config = config.model_copy(update={"max_completions": 1})
    except ValidationError:
        report.update(status="not_run", error_reason="invalid_query_configuration")
        return report, 2
    context = ReaderContext(config.context_tokens)
    budget = RequestBudget(config=config)
    report.update(context_tokens=context.context_tokens, context_method=context.method,
                  max_new_completions=1)
    phase = "preflight"
    capture = {}
    try:
        system, user = build_planner_prompt(question)
        report["planner_prompt_tokens"] = context.check(system, user, config)
        references = (resolver or resolve_article_references)(question)
        phase = "completion"
        captured_client = _CapturedClient(llm._client if client is None else client, capture)
        proposal = (completion or structured_completion)(
            system, user, schema=PlannerOutput, budget=budget, client=captured_client,
        )
        groups = (("facts", [fact.quote for fact in proposal.facts]),
                  ("reference_quotes", proposal.reference_quotes),
                  ("unprocessed_parts", proposal.unprocessed_parts))
        for category, quotes in groups:
            for index, quote in enumerate(quotes):
                item = {"category": category, "index": index, "quote_length": len(quote),
                        "matches": sum(question.startswith(quote, position)
                                       for position in range(len(question) + 1))}
                if show_quotes:
                    item["quote"] = quote
                report["quote_checks"].append(item)
        phase = "validation"
        (validator or validate_plan)(question, proposal, budget=budget, references=references)
        report["status"] = "valid"
        return report, 0
    except QueryPipelineError as error:
        report["status"] = phase + "_failed"
        report["validation_reason" if phase == "validation" else "error_reason"] = error.reason
        return report, 1
    finally:
        report["new_completion_calls"] = budget.completion_calls
        if phase != "preflight":
            parsed, status, errors = _inspect_response(capture, PlannerOutput)
            report.update(parsed_plan_status=status, schema_errors=errors)
            if show_plan:
                report["planner_json"] = parsed


def main(argv=None, *, completion=None, validator=None, resolver=None, client=None) -> int:
    parser = argparse.ArgumentParser(
        description="Send at most one planner completion and diagnose its original quote provenance. "
                    "No retrieval, reranker or answer-reader work. A new proposal can differ from a prior failure.",
        epilog="Exit codes:0 plan validated;1 expected context/completion/validation failure;"
               "2 usage/config/output error. Default output omits question, quotes and full proposals.",
    )
    parser.add_argument("--question", required=True, help="Exact original question, including punctuation and spacing.")
    parser.add_argument("--context-tokens", type=_positive_integer, help="Confirmed reader context override for this run only.")
    parser.add_argument("--output", type=Path, help="New JSON file, created exclusively; existing paths are refused before any request.")
    parser.add_argument("--show-quotes", action="store_true", help="Explicitly include proposed quote text in the report.")
    parser.add_argument("--show-plan", action="store_true",
                        help="Include captured planner JSON, including quotes, queries and scopes; may disclose question-derived/private content. No extra completion.")
    args = parser.parse_args(argv)
    output = None
    try:
        if args.output is not None:
            try:
                output = args.output.open("x", encoding="utf-8")
            except OSError as error:
                report = _metadata(args.question)
                report.update(status="not_run", error_reason="output_exists" if isinstance(error, FileExistsError) else "output_unavailable")
                print(json.dumps(report, ensure_ascii=False))
                return 2
        _load_environment()
        report, code = diagnose(args.question, context_tokens=args.context_tokens, show_quotes=args.show_quotes, show_plan=args.show_plan,
                                completion=completion, validator=validator, resolver=resolver, client=client)
        encoded = json.dumps(report, ensure_ascii=False, indent=2)
        if output is not None:
            try:
                output.write(encoded + "\n")
                output.flush()
            except OSError:
                report.update(status="output_failed", error_reason="output_write_failed")
                print(json.dumps(report, ensure_ascii=False))
                return 2
        print(encoded)
        return code
    finally:
        if output is not None:
            output.close()


if __name__ == "__main__":
    raise SystemExit(main())
