"""Offline-safe preflight and user-run evaluation runner for the query pipeline.

Importing this module never loads embedding/reranker models, constructs network
clients, or executes inference. Real suites run only on explicit CLI flags.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from copy import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
import hashlib
import json
import os
import math
import re
from pathlib import Path
import struct
import sys
import time
from typing import Any
from types import SimpleNamespace
import urllib.parse
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ragdn.config import CHUNKS_PATH, EMB_PATH, MANIFEST_PATH, META_PATH, MODEL_EMBEDDING, MODEL_RERANKER
from ragdn.query_config import QueryConfig
from ragdn.query_evaluation import (
    aggregate_case_reports,
    corpus_semantic_hash,
    evaluate_case,
    evaluate_suite,
    evidence_coverage,
    quantiles,
    retrieval_at_k,
    summarize_observations,
    validate_smoke_artifact,
)

SMOKE_PATH = ROOT / "eval" / "query_pipeline_smoke.json"
GOLDEN_PATH = ROOT / "eval" / "golden_dev_test_retrieval.jsonl"
DEFAULT_OUTPUT_ROOT = ROOT / "eval" / "runs" / "query_pipeline"
REQUIRED_DEPENDENCIES = (
    "numpy", "pydantic", "yaml", "httpx", "openai",
    "sentence_transformers", "rank_bm25", "fastapi", "dotenv",
)


def _runtime_environment(environ: Mapping[str, str] | None) -> dict[str, str]:
    if environ is not None:
        return dict(environ)
    # Match llm.py's .env precedence without mutating the process or disclosing
    # credentials. Explicit environment values still win over file defaults.
    values: dict[str, str] = {}
    if importlib.util.find_spec("dotenv") is not None:
        from dotenv import dotenv_values
        values = {key: value for key, value in dotenv_values(ROOT / ".env").items() if value is not None}
    return {**values, **os.environ}


def _file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _cached_model_available(model: str, env: Mapping[str, str]) -> bool:
    home = Path(env.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    cache = Path(env.get("HF_HUB_CACHE", env.get("HUGGINGFACE_HUB_CACHE", str(home / "hub"))))
    if env.get("SENTENCE_TRANSFORMERS_HOME"):
        cache = Path(env["SENTENCE_TRANSFORMERS_HOME"])
    snapshots = cache / ("models--" + model.replace("/", "--")) / "snapshots"
    if not snapshots.is_dir():
        return False
    for snapshot in snapshots.iterdir():
        if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
            continue
        if not any((snapshot / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json")):
            continue
        if any((snapshot / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
            return True
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index = snapshot / name
            if index.is_file():
                try:
                    shards = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
                except (OSError, ValueError):
                    continue
                if isinstance(shards, Mapping) and shards and all(
                    isinstance(part, str) and (snapshot / part).is_file() for part in shards.values()
                ):
                    return True
    return False


@contextmanager
def _cache_only_model_loading():
    """Temporary process flags prevent implicit HF downloads during user runs."""
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update(dict.fromkeys(names, "1"))
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _sanitize_endpoint(base_url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlsplit(base_url or "http://localhost:11434/v1")
    scheme = parsed.scheme if parsed.scheme in {"http", "https"} else "http"
    host = parsed.hostname or "localhost"
    host = f"[{host}]" if ":" in host else host
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    clean_base = urllib.parse.urlunsplit((scheme, netloc, parsed.path or "/v1", "", ""))
    ps_url = urllib.parse.urlunsplit((scheme, netloc, "/api/ps", "", ""))
    return clean_base, ps_url


def _model_matches(configured: str, candidate: str | None) -> bool:
    if not isinstance(candidate, str) or not candidate.strip():
        return False
    left, right = configured.strip().lower(), candidate.strip().lower()
    return left == right or f"{left}:latest" == right or left == f"{right}:latest"


def fetch_ollama_ps(
    ps_url: str, *, timeout: float = 5.0, opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    """Read-only /api/ps probe; never warms up models or logs upstream bodies."""
    try:
        with opener(ps_url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        return {"status": "unreachable", "error_type": type(exc).__name__, "running_models": []}
    raw_models = payload.get("models") if isinstance(payload, Mapping) else []
    models = []
    if isinstance(raw_models, list):
        for item in raw_models:
            if isinstance(item, Mapping):
                ctx = item.get("context_length")
                models.append({
                    "name": item.get("name") if isinstance(item.get("name"), str) else None,
                    "model": item.get("model") if isinstance(item.get("model"), str) else None,
                    "context_length": ctx if type(ctx) is int and ctx > 0 else None,
                })
    return {"status": "reachable", "running_models": models}


def _npy_row_count(path: Path) -> tuple[int, ...]:
    with path.open("rb") as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError("invalid_npy_magic")
        major, _ = struct.unpack("BB", handle.read(2))
        if major not in {1, 2, 3}:
            raise ValueError("unsupported_npy_version")
        header_len = struct.unpack("<H" if major == 1 else "<I", handle.read(2 if major == 1 else 4))[0]
        if header_len > 1024 * 1024:
            raise ValueError("invalid_npy_header_length")
        header = ast.literal_eval(handle.read(header_len).decode("latin1").strip())
        data_offset = handle.tell()
    shape = header.get("shape") if isinstance(header, Mapping) else None
    if not isinstance(shape, tuple) or not shape or any(type(dim) is not int or dim < 0 for dim in shape):
        raise ValueError("invalid_npy_shape")
    descriptor = header.get("descr")
    if not isinstance(descriptor, str) or descriptor not in {"<f2", ">f2", "<f4", ">f4", "<f8", ">f8"}:
        raise ValueError("unsupported_embedding_dtype")
    if path.stat().st_size != data_offset + math.prod(shape) * int(descriptor[-1]):
        raise ValueError("incomplete_embedding_file")
    return shape


def run_preflight(
    *,
    manifest_path: Path = MANIFEST_PATH,
    chunks_path: Path = CHUNKS_PATH,
    emb_path: Path = EMB_PATH,
    meta_path: Path = META_PATH,
    smoke_path: Path = SMOKE_PATH,
    golden_path: Path = GOLDEN_PATH,
    environ: Mapping[str, str] | None = None,
    spec_finder: Callable[[str], Any] = importlib.util.find_spec,
    ps_fetcher: Callable[[str], Mapping[str, Any]] | None = None,
    probe_ollama: bool = True,
    require_context: bool = True,
    model_cache_checker: Callable[[str, Mapping[str, str]], bool] = _cached_model_available,
) -> dict[str, Any]:
    """Verify dependencies, corpus/index/smoke hashes, and confirmed reader context."""
    env = _runtime_environment(environ)
    errors: list[str] = []
    warnings: list[str] = []
    guidance: list[str] = []

    missing_deps = [name for name in REQUIRED_DEPENDENCIES if spec_finder(name) is None]
    if missing_deps:
        errors.append(f"missing_dependencies:{','.join(sorted(missing_deps))}")

    config_info: dict[str, Any] = {}
    configured_context: int | None = None
    try:
        qcfg = QueryConfig.from_env(env)
        configured_context = qcfg.context_tokens
        config_info = qcfg.model_dump()
    except ValueError as exc:
        errors.append(f"invalid_query_config:{type(exc).__name__}")

    if not manifest_path.exists():
        errors.append("missing_manifest_file")
    elif "yaml" not in missing_deps:
        try:
            import yaml
            manifest_data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest_data, Mapping) or not isinstance(manifest_data.get("documents"), list):
                errors.append("invalid_manifest_structure")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            errors.append(f"corrupt_manifest_file:{type(exc).__name__}")
    chunks: list[dict[str, Any]] = []
    semantic_hash: str | None = None
    if not chunks_path.exists():
        errors.append("missing_chunks_file")
    else:
        try:
            chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not chunks:
                errors.append("empty_chunks_file")
            elif len({row.get("chunk_id") for row in chunks}) != len(chunks):
                errors.append("duplicate_chunk_ids")
            else:
                semantic_hash = corpus_semantic_hash(chunks)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            errors.append(f"corrupt_chunks_file:{type(exc).__name__}")

    index_info: dict[str, Any] = {"emb_exists": emb_path.exists(), "meta_exists": meta_path.exists()}
    if not emb_path.exists() or not meta_path.exists():
        errors.append("missing_index_files")
    else:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            shape = _npy_row_count(emb_path)
            index_info.update({
                "meta_n": meta.get("n"),
                "emb_rows": shape[0],
                "emb_dim": shape[1] if len(shape) > 1 else None,
                "content_sha256": meta.get("content_sha256"),
                "embedding_model": meta.get("model"),
            })
            if chunks and (meta.get("n") != len(chunks) or shape[0] != len(chunks)):
                errors.append("index_count_mismatch")
            if semantic_hash and meta.get("content_sha256") != semantic_hash:
                errors.append("index_content_hash_mismatch")
            if meta.get("model") != MODEL_EMBEDDING:
                warnings.append("index_embedding_model_mismatch")
        except (OSError, ValueError, TypeError, AttributeError, SyntaxError, struct.error) as exc:
            errors.append(f"corrupt_index_files:{type(exc).__name__}")

    artifact = None
    smoke_info: dict[str, Any] = {"exists": smoke_path.exists()}
    if not smoke_path.exists():
        errors.append("missing_smoke_artifact")
    elif chunks:
        try:
            artifact = json.loads(smoke_path.read_text(encoding="utf-8"))
            val = validate_smoke_artifact(artifact, chunks)
            smoke_info.update({
                "status": val.get("status"),
                "annotation_status": val.get("annotation_status"),
                "case_count": len(val.get("cases", {})),
                "locator_count": sum(row.get("locator_count", 0) for row in val.get("cases", {}).values()),
                "errors": val.get("errors", []),
            })
            if val.get("status") != "valid":
                errors.append("invalid_smoke_artifact")
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            errors.append(f"corrupt_smoke_artifact:{type(exc).__name__}")

    golden_info: dict[str, Any] = {"exists": golden_path.exists(), "count": 0}
    if not golden_path.exists():
        errors.append("missing_golden_dataset")
    else:
        try:
            rows = [json.loads(line) for line in golden_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            golden_info["count"] = len(rows)
            if not rows:
                errors.append("empty_golden_dataset")
            if isinstance(artifact, Mapping):
                known = {row["id"]: row for row in rows if isinstance(row, Mapping) and isinstance(row.get("id"), str)}
                copied = [case for case in artifact.get("cases", []) if case.get("origin") == "existing_golden_copy"]
                smoke_info["golden_question_copies_validated"] = sum(
                    case.get("question") == known.get(case.get("source_golden_id"), {}).get("question")
                    and case.get("source_golden_id") in known for case in copied)
                if smoke_info["golden_question_copies_validated"] != len(copied):
                    errors.append("smoke_golden_question_mismatch")
        except (OSError, ValueError) as exc:
            errors.append(f"corrupt_golden_dataset:{type(exc).__name__}")

    configured_model = env.get("LLM_MODEL", "qwen2.5:3b").strip() or "qwen2.5:3b"
    try:
        clean_base, ps_url = _sanitize_endpoint(env.get("LLM_BASE_URL", "http://localhost:11434/v1"))
    except ValueError:
        clean_base, ps_url = None, None
        errors.append("invalid_reader_endpoint")
    ps_result: Mapping[str, Any] = {"status": "not_queried", "running_models": []}
    if probe_ollama and ps_url is not None:
        fetcher = fetch_ollama_ps if ps_fetcher is None else ps_fetcher
        ps_result = fetcher(ps_url)

    loaded_context: int | None = None
    for item in ps_result.get("running_models", []):
        if isinstance(item, Mapping) and (
            _model_matches(configured_model, item.get("name"))
            or _model_matches(configured_model, item.get("model"))
        ):
            ctx = item.get("context_length")
            if type(ctx) is int and ctx > 0:
                loaded_context = ctx
                break

    confirmed_context = configured_context if configured_context is not None else loaded_context
    context_source = (
        "configured_env" if configured_context is not None
        else "ollama_api_ps" if loaded_context is not None
        else "unknown_requires_confirmation"
    )
    if confirmed_context is None:
        guidance.append(
            "Reader context is unconfirmed. Set RAG_QUERY_CONTEXT_TOKENS=<positive_int> "
            "or ensure the configured LLM_MODEL is loaded in Ollama (/api/ps) before adaptive runs."
        )
        if require_context:
            errors.append("context_budget_unconfirmed")

    model_cache = {name: bool(model_cache_checker(name, env)) for name in (MODEL_EMBEDDING, MODEL_RERANKER)}
    if not all(model_cache.values()):
        errors.append("local_model_cache_incomplete")
        guidance.append("Embedding/reranker cache is incomplete; runner will not download models. Prepare local models separately before an explicit suite.")
    fingerprints = {}
    for name, path in (("manifest", manifest_path), ("chunks", chunks_path), ("embeddings", emb_path),
                       ("index_metadata", meta_path), ("smoke_annotation", smoke_path), ("golden", golden_path)):
        if path.is_file():
            try:
                fingerprints[name] = _file_fingerprint(path)
            except OSError:
                errors.append(f"unreadable_fingerprint:{name}")
    config_fingerprint = {**config_info, "embedding_model": MODEL_EMBEDDING,
                          "reranker_model": MODEL_RERANKER, "reader_model": configured_model,
                          "confirmed_context_tokens": confirmed_context}
    fingerprints["query_config_and_model_identifiers"] = {
        "sha256": hashlib.sha256(json.dumps(config_fingerprint, sort_keys=True).encode("utf-8")).hexdigest(),
        "basis": "validated_config_and_identifiers_not_model_weight_bytes",
    }

    return {
        "status": "ready" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "guidance": guidance,
        "dependencies": {"required": list(REQUIRED_DEPENDENCIES), "missing": missing_deps},
        "corpus": {"chunk_count": len(chunks), "corpus_sha256": semantic_hash},
        "index": index_info,
        "smoke_artifact": smoke_info,
        "golden_dataset": golden_info,
        "fingerprints": fingerprints,
        "runtime": {
            "embedding_model": MODEL_EMBEDDING,
            "reranker_model": MODEL_RERANKER,
            "reader_model": configured_model,
            "base_url": clean_base,
            "ollama_ps_url": ps_url,
            "ollama_ps_status": ps_result.get("status"),
            "configured_context_tokens": configured_context,
            "loaded_context_tokens": loaded_context,
            "confirmed_context_tokens": confirmed_context,
            "context_source": context_source,
            "token_accounting_method": "conservative_utf8_bytes",
            "query_config": config_info,
            "local_model_cache": model_cache,
            "model_loading": "cache_only_no_implicit_download",
        },
    }


_JSON_FAILURE_MARKERS = frozenset({
    "invalid_json", "schema_validation_failed", "empty_completion", "length_truncated",
})


def create_run_directory(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    run_id: str | None = None,
    now: datetime | None = None,
) -> Path:
    """Create a new unique run directory; refuse collision or protected targets."""
    resolved_root = output_root.resolve()
    protected = {
        (ROOT / "eval" / "golden_check.jsonl").resolve(),
        SMOKE_PATH.resolve(),
        GOLDEN_PATH.resolve(),
    }
    if resolved_root in protected:
        raise ValueError("output_root cannot target a protected evaluation file")
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    folder = run_id if run_id else f"{stamp}_{uuid.uuid4().hex[:8]}"
    if not isinstance(folder, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", folder):
        raise ValueError("run_id must be a single safe directory name")
    run_dir = output_root / folder
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_report(run_dir: Path, report: Mapping[str, Any], filename: str = "report.json") -> Path:
    if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.json", filename):
        raise ValueError("report filename must be a safe JSON basename")
    report_path = run_dir / filename
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report_path


def observation_from_query_result(
    result: Any,
    chunks: Sequence[Mapping[str, Any]],
    *,
    is_legacy_branch: bool = False,
    answer_payload: Mapping[str, Any] | None = None,
    answer_judgments: Mapping[str, bool] | None = None,
    execution_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a P4 QueryResult into a P6-A observation plus content-free trace."""
    parents = {row["chunk_id"]: row for row in chunks}
    sources = list(getattr(result, "sources", ()) or ())
    if not sources and isinstance(getattr(result, "retrieval", None), Mapping):
        hits = result.retrieval.get("hits", [])
        sources = [chunks[idx] for idx in hits if 0 <= idx < len(chunks)]

    retrieved_articles = [f"{row['doc_id']}:{row['dieu']}" for row in sources]
    context_spans: list[dict[str, Any]] = []
    views = list(getattr(result, "context_views", ()) or ())
    for view in views:
        parent = parents.get(view.parent_chunk_id)
        if parent is None:
            continue
        for span in view.spans:
            context_spans.append({
                "parent_chunk_id": view.parent_chunk_id,
                "doc_id": parent["doc_id"],
                "dieu": parent["dieu"],
                "start": int(span.start),
                "end": int(span.end),
                "quote": parent["text"][int(span.start):int(span.end)],
            })

    proc = result.processing
    budget = proc.budget
    diag = getattr(result, "diagnostics", {}) or {}
    codes = [str(proc.route_reason), *(str(w) for w in proc.warnings)]
    partial_reasons = [str(r) for r in proc.partial_reasons]

    is_fallback = False if (is_legacy_branch or str(proc.mode) == "legacy") else (str(proc.mode) == "baseline_fallback")
    invalid_json = any(
        code in _JSON_FAILURE_MARKERS or "json" in code.lower() or "schema" in code.lower()
        for code in codes
    )
    budget_exhausted = bool(
        "budget_exhausted" in partial_reasons
        or budget.not_processed
        or any("budget_exhausted" in code or "deadline_exhausted" in code for code in codes)
    )
    estimates = diag.get("prompt_tokens") if isinstance(diag.get("prompt_tokens"), Mapping) else {}
    captured = execution_metrics or {}
    prompt_tokens_map = captured.get("prompt_tokens", {})
    total_prompt_tokens = sum(prompt_tokens_map.values()) if execution_metrics is not None else None
    stage_seconds = {
        str(k): float(v)
        for k, v in (diag.get("stage_seconds") or {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    stage_seconds.update(captured.get("stage_seconds", {}))

    trace: dict[str, Any] = {
        "trace_id": proc.trace_id or diag.get("trace_id"),
        "mode": str(proc.mode),
        "route_reason": str(proc.route_reason),
        "status": str(proc.status),
        "partial": bool(getattr(result, "partial", str(proc.status) == "partial")),
        "partial_reasons": partial_reasons,
        "warnings": list(proc.warnings),
        "selected_source_ids": [row["chunk_id"] for row in sources],
        "seen_spans": {
            view.parent_chunk_id: [[int(s.start), int(s.end)] for s in view.spans]
            for view in views
        },
        "prompt_tokens_by_stage": {str(k): int(v) for k, v in prompt_tokens_map.items()},
        "packing_estimates_by_stage": {str(k): int(v) for k, v in estimates.items()},
        "token_accounting_method": "conservative_utf8_bytes",
        "completion_call_scope": "structured_stages_only",
        "bounded_rerank_pairs": int(budget.initial_rerank_pairs + budget.supplementary_rerank_pairs),
    }
    if answer_payload is not None:
        reader = (answer_payload.get("processing") or {}).get("reader_context") or {}
        reader_prompts = reader.get("prompt_tokens")
        reader_calls = reader.get("reader_calls")
        if isinstance(reader_prompts, Mapping) and type(reader_calls) is int:
            trace["reader_context"] = dict(reader)
            same_method = reader.get("method") == "conservative_utf8_bytes"
            total_prompt_tokens = (total_prompt_tokens + sum(reader_prompts.values())
                                   if total_prompt_tokens is not None and same_method else None)
            if not same_method:
                trace["context_accounting_warning"] = "mixed_methods_reported_separately"
            trace["total_completion_calls"] = int(budget.completion_calls) + reader_calls
        else:
            total_prompt_tokens = None
            trace["reader_context"] = {"status": "not_assessed", "reason": "missing_reader_measurements"}
            trace["total_completion_calls"] = None
        trace["answer_execution"] = {
            "executed": True,
            "refused": bool(answer_payload.get("refused", False)),
            "partial": bool(answer_payload.get("partial", False)),
            "source_count": len(answer_payload.get("sources", []) or []),
            "quality_assessment": "not_assessed_requires_supplied_judgments",
        }

    return {
        "retrieved_articles": retrieved_articles,
        "context_spans": context_spans,
        "fallback": is_fallback,
        "invalid_json": invalid_json,
        "budget_exhausted": budget_exhausted,
        "completion_calls": int(budget.completion_calls),
        "rerank_pairs": captured.get("rerank_pairs") if execution_metrics is not None else None,
        "context_tokens": total_prompt_tokens,
        "stage_seconds": stage_seconds,
        "answer_judgments": answer_judgments,
        "trace": trace,
    }


class _CapturedClient:
    """Eval-only transport observer; stores counts/timings, never prompt bodies."""
    def __init__(self, client, metrics: dict[str, Any]):
        self._client, self._metrics = client, metrics
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def with_options(self, **options):
        return _CapturedClient(self._client.with_options(**options), self._metrics)

    def _create(self, **kwargs):
        messages = kwargs.get("messages", [])
        payload = json.loads(messages[-1]["content"]) if messages else {}
        stage = ("recheck" if "previous_assessments" in payload else
                 "mapper" if "sources" in payload else "planner")
        used = sum(len(message.get("content", "").encode("utf-8")) for message in messages)
        tokens = self._metrics.setdefault("prompt_tokens", {})
        tokens[stage] = tokens.get(stage, 0) + used
        started = time.monotonic()
        try:
            return self._client.chat.completions.create(**kwargs)
        finally:
            seconds = self._metrics.setdefault("stage_seconds", {})
            seconds[stage] = seconds.get(stage, 0.0) + max(0.0, time.monotonic() - started)


class _CapturedReranker:
    def __init__(self, reranker, metrics):
        self._reranker, self._metrics = reranker, metrics

    def __getattr__(self, name):
        return getattr(self._reranker, name)

    def predict(self, pairs, *args, **kwargs):
        self._metrics["rerank_pairs"] += len(pairs)
        started = time.monotonic()
        try:
            return self._reranker.predict(pairs, *args, **kwargs)
        finally:
            seconds = self._metrics.setdefault("stage_seconds", {})
            seconds["rerank"] = seconds.get("rerank", 0.0) + max(0.0, time.monotonic() - started)


def compute_approved_golden_retrieval(
    golden_rows: Sequence[Mapping[str, Any]],
    retrieved_by_id: Mapping[str, Sequence[Any] | None],
    *,
    k: int = 5,
    basis: str = "approved_existing_golden_article_labels_independent_of_draft_span_annotations",
) -> dict[str, Any]:
    """Compute legacy same-k Hit/Recall/MRR from approved golden article labels."""
    buckets: dict[str, list[dict[str, Any]]] = {"ALL": []}
    missing_count = 0
    refusal_skipped = 0
    for row in golden_rows:
        if row.get("should_refuse", False):
            refusal_skipped += 1
            continue
        gold = row.get("gold") or row.get("expected_articles") or []
        if not gold:
            continue
        case_id = str(row.get("id") or row.get("case_id"))
        retrieved = retrieved_by_id.get(case_id)
        metric = retrieval_at_k(retrieved, gold, k=k)
        if metric["status"] != "assessed":
            missing_count += 1
            continue
        buckets["ALL"].append(metric)
        bucket_name = str(row.get("type") or row.get("category") or "normal")
        buckets.setdefault(bucket_name, []).append(metric)

    summary_buckets: dict[str, Any] = {}
    for name, items in buckets.items():
        if not items:
            summary_buckets[name] = {
                "status": "not_assessed",
                "k": k,
                "n": 0,
                "not_assessed": missing_count if name == "ALL" else None,
                f"hit@{k}": None,
                f"recall@{k}": None,
                f"mrr@{k}": None,
            }
        else:
            count = len(items)
            summary_buckets[name] = {
                "status": "assessed",
                "k": k,
                "n": count,
                "not_assessed": missing_count if name == "ALL" else 0,
                f"hit@{k}": sum(x["hit"] for x in items) / count,
                f"recall@{k}": sum(x["recall"] for x in items) / count,
                f"mrr@{k}": sum(x["mrr"] for x in items) / count,
            }
    return {
        "basis": basis,
        "k": k,
        "refusal_cases_excluded": refusal_skipped,
        "buckets": summary_buckets,
    }


def _approved_smoke_labels(smoke_artifact, golden_rows, retrieved_by_id):
    questions = {str(case["case_id"]): case.get("question") for case in smoke_artifact.get("cases", [])}
    candidates = [row for row in golden_rows if str(row.get("id")) in retrieved_by_id]
    mismatched = [str(row["id"]) for row in candidates if row.get("question") != questions.get(str(row["id"]))]
    return [row for row in candidates if str(row["id"]) not in mismatched], mismatched


def _sanitize_observations_for_output(
    observations: Mapping[str, Mapping[str, Any]], *, debug_content: bool = False,
) -> dict[str, dict[str, Any]]:
    sanitized: dict[str, dict[str, Any]] = {}
    for case_id, obs in observations.items():
        spans = []
        for span in obs.get("context_spans") or []:
            entry = {
                "parent_chunk_id": span["parent_chunk_id"],
                "doc_id": span["doc_id"],
                "dieu": span["dieu"],
                "start": span["start"],
                "end": span["end"],
            }
            if debug_content and "quote" in span:
                entry["quote"] = span["quote"]
            spans.append(entry)
        sanitized[case_id] = {
            "retrieved_articles": list(obs.get("retrieved_articles") or []),
            "context_spans": spans,
            "fallback": obs.get("fallback"),
            "invalid_json": obs.get("invalid_json"),
            "budget_exhausted": obs.get("budget_exhausted"),
            "completion_calls": obs.get("completion_calls"),
            "rerank_pairs": obs.get("rerank_pairs"),
            "context_tokens": obs.get("context_tokens"),
            "stage_seconds": dict(obs.get("stage_seconds") or {}),
            "trace": dict(obs.get("trace") or {}),
        }
    return sanitized


def _run_branch_case(
    question: str,
    retriever: Any,
    *,
    branch: str,
    config: QueryConfig,
    context_tokens: int | None,
    manifest: Any = None,
    client: Any = None,
    model: str | None = None,
    cached_plan: Any = None,
    pipeline_runner: Callable[..., Any] | None = None,
    execution_metrics: dict[str, Any] | None = None,
) -> Any:
    from ragdn.query_pipeline import ReaderContext, run_query_pipeline

    runner = run_query_pipeline if pipeline_runner is None else pipeline_runner
    if execution_metrics is not None:
        # A request-local shallow clone shares unchanged model/source objects;
        # its observer counts legacy/fallback predict calls as well as windows.
        observed = copy(retriever)
        execution_metrics["rerank_pairs"] = 0
        observed.reranker = _CapturedReranker(retriever.reranker, execution_metrics)
        retriever = observed
    if branch != "A" and execution_metrics is not None:
        if client is None:
            from ragdn.llm import _client
            client = _client
        client = _CapturedClient(client, execution_metrics)
    reader_ctx = ReaderContext(context_tokens=context_tokens)
    if branch == "A":
        legacy_cfg = config.model_copy(update={"mode": "legacy"})
        return runner(
            question, retriever, config=legacy_cfg, context=reader_ctx,
            manifest=manifest, client=client, model=model, variant="D",
        )
    adaptive_cfg = config.model_copy(update={"mode": "adaptive"})
    return runner(
        question, retriever, config=adaptive_cfg, context=reader_ctx,
        manifest=manifest, client=client, model=model, variant=branch,
        cached_plan=cached_plan if branch == "D" else None,
    )


def execute_smoke_suite(
    smoke_artifact: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    retriever: Any,
    *,
    golden_rows: Sequence[Mapping[str, Any]] = (),
    mode: str = "adaptive",
    with_answer: bool = False,
    k: int = 5,
    config: QueryConfig | None = None,
    context_tokens: int | None = None,
    manifest: Any = None,
    client: Any = None,
    model: str | None = None,
    pipeline_runner: Callable[..., Any] | None = None,
    answer_runner: Callable[..., Mapping[str, Any]] | None = None,
    answer_judgments_by_id: Mapping[str, Mapping[str, bool]] | None = None,
    debug_content: bool = False,
) -> dict[str, Any]:
    if mode not in {"legacy", "adaptive"}:
        raise ValueError("mode must be 'legacy' or 'adaptive'")
    base_cfg = config or QueryConfig(mode=mode, context_tokens=context_tokens)
    eff_ctx = context_tokens if context_tokens is not None else base_cfg.context_tokens
    branch = "A" if mode == "legacy" else "D"

    observations: dict[str, dict[str, Any]] = {}
    retrieved_by_id: dict[str, list[str]] = {}
    for case in smoke_artifact.get("cases", []):
        cid = str(case["case_id"])
        question = str(case["question"])
        captured: dict[str, Any] = {}
        started = time.monotonic()
        qres = _run_branch_case(
            question, retriever, branch=branch, config=base_cfg, context_tokens=eff_ctx,
            manifest=manifest, client=client, model=model, pipeline_runner=pipeline_runner,
            execution_metrics=captured,
        )
        captured.setdefault("stage_seconds", {})["retrieval_total"] = time.monotonic() - started
        ans_payload: Mapping[str, Any] | None = None
        if with_answer:
            started = time.monotonic()
            eff_ans_runner = answer_runner or _default_answer_runner(
                retriever, manifest, config=base_cfg, context_tokens=eff_ctx,
            )
            ans_payload = eff_ans_runner(question, query_result=qres, mode=mode)
            captured["stage_seconds"]["answer_total"] = time.monotonic() - started
        judgments = (answer_judgments_by_id or {}).get(cid)
        obs = observation_from_query_result(
            qres, chunks, is_legacy_branch=(branch == "A"),
            answer_payload=ans_payload, answer_judgments=judgments,
            execution_metrics=captured,
        )
        observations[cid] = obs
        retrieved_by_id[cid] = obs["retrieved_articles"]

    suite_eval = evaluate_suite(smoke_artifact, observations, chunks, k=k)
    approved_subset, label_mismatches = _approved_smoke_labels(smoke_artifact, golden_rows, retrieved_by_id)
    legacy_retrieval = compute_approved_golden_retrieval(
        approved_subset, retrieved_by_id, k=k,
        basis="approved_existing_golden_article_labels_independent_of_draft_span_annotations",
    )
    smoke_diagnostic_retrieval = compute_approved_golden_retrieval(
        [c for c in smoke_artifact.get("cases", []) if c.get("expected_articles")],
        retrieved_by_id, k=k,
        basis="smoke_declared_expected_articles_diagnostic_only_not_approved",
    )
    return {
        "suite": "smoke",
        "mode": mode,
        "with_answer": with_answer,
        "k": k,
        "legacy_article_retrieval": legacy_retrieval,
        "golden_label_question_mismatch_case_ids": label_mismatches,
        "smoke_declared_article_diagnostics": smoke_diagnostic_retrieval,
        "smoke_annotation_evaluation": suite_eval,
        "case_observations": _sanitize_observations_for_output(observations, debug_content=debug_content),
        "acceptance": "user_review_required",
        "automated_ab_gate": None,
    }


def execute_golden_suite(
    golden_rows: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    retriever: Any,
    *,
    mode: str = "legacy",
    with_answer: bool = False,
    k: int = 5,
    config: QueryConfig | None = None,
    context_tokens: int | None = None,
    manifest: Any = None,
    client: Any = None,
    model: str | None = None,
    pipeline_runner: Callable[..., Any] | None = None,
    answer_runner: Callable[..., Mapping[str, Any]] | None = None,
    debug_content: bool = False,
) -> dict[str, Any]:
    if mode not in {"legacy", "adaptive"}:
        raise ValueError("mode must be 'legacy' or 'adaptive'")
    base_cfg = config or QueryConfig(mode=mode, context_tokens=context_tokens)
    eff_ctx = context_tokens if context_tokens is not None else base_cfg.context_tokens
    branch = "A" if mode == "legacy" else "D"

    observations: dict[str, dict[str, Any]] = {}
    retrieved_by_id: dict[str, list[str]] = {}
    for row in golden_rows:
        cid = str(row["id"])
        question = str(row["question"])
        captured: dict[str, Any] = {}
        started = time.monotonic()
        qres = _run_branch_case(
            question, retriever, branch=branch, config=base_cfg, context_tokens=eff_ctx,
            manifest=manifest, client=client, model=model, pipeline_runner=pipeline_runner,
            execution_metrics=captured,
        )
        captured.setdefault("stage_seconds", {})["retrieval_total"] = time.monotonic() - started
        ans_payload: Mapping[str, Any] | None = None
        if with_answer:
            started = time.monotonic()
            eff_ans_runner = answer_runner or _default_answer_runner(
                retriever, manifest, config=base_cfg, context_tokens=eff_ctx,
            )
            ans_payload = eff_ans_runner(question, query_result=qres, mode=mode)
            captured["stage_seconds"]["answer_total"] = time.monotonic() - started
        obs = observation_from_query_result(
            qres, chunks, is_legacy_branch=(branch == "A"), answer_payload=ans_payload,
            execution_metrics=captured,
        )
        observations[cid] = obs
        retrieved_by_id[cid] = obs["retrieved_articles"]

    return {
        "suite": "golden",
        "mode": mode,
        "with_answer": with_answer,
        "k": k,
        "legacy_article_retrieval": compute_approved_golden_retrieval(golden_rows, retrieved_by_id, k=k),
        "operational_summary": summarize_observations(list(observations.values())),
        "case_observations": _sanitize_observations_for_output(observations, debug_content=debug_content),
        "acceptance": "user_review_required",
    }


def execute_compare_suite(
    smoke_artifact: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    retriever: Any,
    *,
    golden_rows: Sequence[Mapping[str, Any]] = (),
    with_answer: bool = False,
    k: int = 5,
    config: QueryConfig | None = None,
    context_tokens: int | None = None,
    manifest: Any = None,
    client: Any = None,
    model: str | None = None,
    reuse_planner: bool = True,
    pipeline_runner: Callable[..., Any] | None = None,
    answer_runner: Callable[..., Mapping[str, Any]] | None = None,
    answer_judgments_by_branch: Mapping[str, Mapping[str, Mapping[str, bool]]] | None = None,
    debug_content: bool = False,
) -> dict[str, Any]:
    """Compare branches A/B/C/D with identical B/C/D budgets and optional C->D planner reuse."""
    shared_cfg = (config or QueryConfig(mode="adaptive", context_tokens=context_tokens)).model_copy(
        update={"mode": "adaptive"},
    )
    eff_ctx = context_tokens if context_tokens is not None else shared_cfg.context_tokens
    recorded_plans: dict[str, Any] = {}
    reused_in_d: list[str] = []
    branches_report: dict[str, Any] = {}

    for branch in ("A", "B", "C", "D"):
        observations: dict[str, dict[str, Any]] = {}
        retrieved_by_id: dict[str, list[str]] = {}
        branch_judgments = (answer_judgments_by_branch or {}).get(branch) or {}
        for case in smoke_artifact.get("cases", []):
            cid = str(case["case_id"])
            question = str(case["question"])
            cached = recorded_plans.get(cid) if (branch == "D" and reuse_planner) else None
            if branch == "D" and cached is not None:
                reused_in_d.append(cid)
            captured: dict[str, Any] = {}
            started = time.monotonic()
            qres = _run_branch_case(
                question, retriever, branch=branch, config=shared_cfg, context_tokens=eff_ctx,
                manifest=manifest, client=client, model=model, cached_plan=cached,
                pipeline_runner=pipeline_runner,
                execution_metrics=captured,
            )
            captured.setdefault("stage_seconds", {})["retrieval_total"] = time.monotonic() - started
            if branch == "C" and getattr(qres, "plan", None) is not None:
                recorded_plans[cid] = qres.plan
            ans_payload: Mapping[str, Any] | None = None
            if with_answer:
                started = time.monotonic()
                eff_ans_runner = answer_runner or _default_answer_runner(
                    retriever, manifest, config=shared_cfg, context_tokens=eff_ctx,
                )
                ans_payload = eff_ans_runner(
                    question, query_result=qres, mode="legacy" if branch == "A" else "adaptive",
                )
                captured["stage_seconds"]["answer_total"] = time.monotonic() - started
            obs = observation_from_query_result(
                qres, chunks, is_legacy_branch=(branch == "A"),
                answer_payload=ans_payload, answer_judgments=branch_judgments.get(cid),
                execution_metrics=captured,
            )
            observations[cid] = obs
            retrieved_by_id[cid] = obs["retrieved_articles"]

        approved_subset, label_mismatches = _approved_smoke_labels(smoke_artifact, golden_rows, retrieved_by_id)
        branches_report[branch] = {
            "branch": branch,
            "legacy_article_retrieval": compute_approved_golden_retrieval(
                approved_subset, retrieved_by_id, k=k,
                basis="approved_existing_golden_article_labels_independent_of_draft_span_annotations",
            ),
            "golden_label_question_mismatch_case_ids": label_mismatches,
            "smoke_declared_article_diagnostics": compute_approved_golden_retrieval(
                [c for c in smoke_artifact.get("cases", []) if c.get("expected_articles")],
                retrieved_by_id, k=k,
                basis="smoke_declared_expected_articles_diagnostic_only_not_approved",
            ),
            "smoke_annotation_evaluation": evaluate_suite(smoke_artifact, observations, chunks, k=k),
            "case_observations": _sanitize_observations_for_output(observations, debug_content=debug_content),
        }

    return {
        "suite": "compare",
        "with_answer": with_answer,
        "k": k,
        "shared_bcd_config": shared_cfg.model_dump(),
        "confirmed_context_tokens": eff_ctx,
        "planner_reuse": {
            "enabled": reuse_planner,
            "recorded_from_c_count": len(recorded_plans),
            "supplied_to_d_case_ids": reused_in_d,
            "capture_scope": "request_local_memory_no_annotations",
            "plan_sha256_by_case_id": {case_id: hashlib.sha256(json.dumps(plan.model_dump(),
                sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
                for case_id, plan in recorded_plans.items()},
            **({"debug_plans": {case_id: plan.model_dump() for case_id, plan in recorded_plans.items()}}
               if debug_content else {}),
        },
        "branches": branches_report,
        "acceptance": "user_review_required",
        "automated_ab_gate": None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_query_pipeline.py",
        description="Read-only preflight and explicit evaluation runner for the query pipeline.",
    )
    parser.add_argument("--preflight", action="store_true", help="Run read-only preflight checks without inference.")
    parser.add_argument("--suite", choices=("smoke", "golden", "compare"), help="Explicit evaluation suite to execute.")
    parser.add_argument("--mode", choices=("legacy", "adaptive"), help="Execution mode for smoke or golden suites.")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--retrieval-only", action="store_true", help="Run retrieval/orchestration only (no reader answer).")
    scope.add_argument("--with-answer", action="store_true", help="Include reader answer generation.")
    parser.add_argument("--k", type=int, default=5, help="Top-k article deduplication cutoff (default: 5).")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Parent directory for run outputs.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional explicit run folder name (fails on collision).")
    parser.add_argument("--no-ollama-ps", action="store_true", help="Skip read-only /api/ps probe during preflight.")
    parser.add_argument("--no-reuse-planner", action="store_true", help="Disable C->D validated planner reuse in compare.")
    parser.add_argument("--debug-content", action="store_true", help="Explicit opt-in to include source quotes in run artifacts.")
    return parser


def _default_answer_runner(
    retriever: Any,
    manifest: Any = None,
    *,
    config: QueryConfig | None = None,
    context_tokens: int | None = None,
) -> Callable[..., Mapping[str, Any]]:
    from ragdn.generate import generate_answer
    from ragdn.query_pipeline import ReaderContext

    def _run(question: str, *, query_result: Any = None, mode: str = "adaptive") -> Mapping[str, Any]:
        eff_ctx = context_tokens if context_tokens is not None else (config.context_tokens if config is not None else None)
        base_cfg = (config or QueryConfig(mode=mode, context_tokens=eff_ctx)).model_copy(
            update={"mode": mode, "context_tokens": eff_ctx},
        )
        reader_ctx = ReaderContext(context_tokens=eff_ctx) if eff_ctx is not None else None
        return generate_answer(
            question,
            retriever,
            query_result=query_result,
            config=base_cfg,
            context=reader_ctx,
            manifest=manifest,
        )

    return _run


def _execute_requested_suite(args, preflight, *, environ, rag_factory, pipeline_runner, answer_runner):
    if rag_factory is None:
        with _cache_only_model_loading():
            from ragdn.pipeline import RAG
            rag_instance = RAG.load()
    else:
        rag_instance = rag_factory()
    chunks = list(rag_instance.retriever.chunks)
    if (len(chunks) != preflight["corpus"]["chunk_count"] or
            corpus_semantic_hash(chunks) != preflight["corpus"]["corpus_sha256"]):
        raise ValueError("loaded_corpus_changed_after_preflight")
    manifest = getattr(rag_instance, "manifest", None)
    confirmed_ctx = preflight["runtime"]["confirmed_context_tokens"]
    qcfg = QueryConfig.from_env(_runtime_environment(environ)).model_copy(update={"context_tokens": confirmed_ctx})
    ans_fn = answer_runner or (
        _default_answer_runner(rag_instance.retriever, manifest, config=qcfg, context_tokens=confirmed_ctx)
        if args.with_answer else None
    )
    common = dict(with_answer=args.with_answer, k=args.k, config=qcfg, context_tokens=confirmed_ctx,
                  manifest=manifest, pipeline_runner=pipeline_runner, answer_runner=ans_fn,
                  debug_content=args.debug_content)
    golden_rows = [json.loads(line) for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.suite == "golden":
        return execute_golden_suite(golden_rows, chunks, rag_instance.retriever, mode=args.mode, **common)
    smoke_artifact = json.loads(SMOKE_PATH.read_text(encoding="utf-8"))
    if args.suite == "smoke":
        return execute_smoke_suite(smoke_artifact, chunks, rag_instance.retriever,
                                   golden_rows=golden_rows, mode=args.mode, **common)
    return execute_compare_suite(smoke_artifact, chunks, rag_instance.retriever,
        golden_rows=golden_rows, reuse_planner=not args.no_reuse_planner, **common)


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    spec_finder: Callable[[str], Any] = importlib.util.find_spec,
    ps_fetcher: Callable[[str], Mapping[str, Any]] | None = None,
    rag_factory: Callable[[], Any] | None = None,
    pipeline_runner: Callable[..., Any] | None = None,
    answer_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    if not args.preflight and args.suite is None:
        print(json.dumps({
            "status": "usage_error",
            "error": "explicit_action_required",
            "message": "Pass --preflight or --suite {smoke,golden,compare} with --retrieval-only|--with-answer.",
        }))
        return 2

    if args.k <= 0:
        print(json.dumps({"status": "usage_error", "error": "invalid_k"}))
        return 2

    if args.preflight and (args.suite is not None or args.retrieval_only or args.with_answer):
        print(json.dumps({"status": "usage_error", "error": "preflight_cannot_execute_suite"}))
        return 2

    if args.suite is not None:
        if not args.retrieval_only and not args.with_answer:
            print(json.dumps({
                "status": "usage_error",
                "error": "missing_execution_depth_flag",
                "message": "Suite execution requires explicit --retrieval-only or --with-answer.",
            }))
            return 2
        if args.suite in {"smoke", "golden"} and args.mode is None:
            print(json.dumps({
                "status": "usage_error",
                "error": "missing_mode_flag",
                "message": f"--suite {args.suite} requires explicit --mode {{legacy,adaptive}}.",
            }))
            return 2
        if args.suite == "compare" and args.mode is not None:
            print(json.dumps({
                "status": "usage_error",
                "error": "compare_disallows_single_mode",
                "message": "--suite compare evaluates internal branches A/B/C/D and does not take --mode.",
            }))
            return 2

    require_ctx = True if args.suite is None else (args.suite == "compare" or args.mode == "adaptive")
    preflight = run_preflight(
        environ=environ,
        spec_finder=spec_finder,
        ps_fetcher=ps_fetcher,
        probe_ollama=not args.no_ollama_ps,
        require_context=require_ctx,
    )
    if args.preflight and args.suite is None:
        print(json.dumps(preflight, ensure_ascii=False))
        return 0 if preflight["status"] == "ready" else 1

    if preflight["status"] != "ready":
        print(json.dumps({"status": "preflight_failed", "preflight": preflight}, ensure_ascii=False))
        return 1

    try:
        run_dir = create_run_directory(args.output_root, run_id=args.run_id)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "output_directory_error", "error_type": type(exc).__name__}))
        return 1

    # Expected I/O/service failures produce a content-free historical report;
    # programming defects still propagate instead of being hidden broadly.
    from httpx import HTTPError
    from openai import APIError
    from ragdn.query_models import QueryPipelineError
    try:
        suite_report = _execute_requested_suite(args, preflight, environ=environ,
            rag_factory=rag_factory, pipeline_runner=pipeline_runner, answer_runner=answer_runner)
    except (OSError, ValueError, HTTPError, APIError, QueryPipelineError) as exc:
        failure = {"schema_version": 1, "status": "failed", "suite": args.suite,
                   "error_type": type(exc).__name__, "preflight": preflight,
                   "acceptance": "user_review_required"}
        report_path = write_run_report(run_dir, failure)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "report_path": str(report_path)}))
        return 1

    full_report = {
        "schema_version": 1,
        "status": "completed",
        "run_directory": str(run_dir),
        "preflight": preflight,
        "evaluation": suite_report,
    }
    report_path = write_run_report(run_dir, full_report)
    print(json.dumps({
        "status": "completed",
        "suite": args.suite,
        "report_path": str(report_path),
        "acceptance": suite_report.get("acceptance"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
