"""Offline contract tests for scripts/check_query_pipeline.py."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from copy import deepcopy
from io import StringIO
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from ragdn.index import build_bm25_indices
from ragdn.query_config import QueryConfig
from ragdn.query_evaluation import corpus_semantic_hash
from ragdn.retrieval import Retriever
from scripts.check_query_pipeline import (
    create_run_directory,
    execute_compare_suite,
    execute_golden_suite,
    execute_smoke_suite,
    main as runner_main,
    run_preflight,
    write_run_report,
    fetch_ollama_ps,
    _cached_model_available,
)
import scripts.check_query_pipeline as runner


class _DummyEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            return np.ones(4, dtype=np.float32) / 2.0
        return np.ones((len(texts), 4), dtype=np.float32) / 2.0


class _DummyTokenizer:
    def __init__(self, specials: int = 3):
        self.specials = specials

    def __call__(self, text, pair=None, *, add_special_tokens=False,
                 truncation=False, return_offsets_mapping=False, verbose=True):
        import re as _re
        assert truncation is False
        spans = [(m.start(), m.end()) for m in _re.finditer(r"\S+", text)]
        ids = list(range(len(spans)))
        if pair is not None:
            ids += list(range(len(_re.findall(r"\S+", pair))))
        if add_special_tokens:
            ids += [999] * self.specials
        res = {"input_ids": ids}
        if return_offsets_mapping:
            res["offset_mapping"] = spans
        return res

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(tok) for tok in token_ids)


class _DummyReranker:
    def __init__(self):
        self.tokenizer = _DummyTokenizer()

    def predict(self, pairs):
        scores = []
        for idx, (_q, passage) in enumerate(pairs):
            scores.append(2.5 if "65%" in passage or "Điều 58" in passage else 0.5 - idx * 0.01)
        return np.array(scores, dtype=np.float32)


class _ScriptedCompletions:
    def __init__(self, handler):
        self._handler = handler
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self._handler(kwargs, len(self.calls))
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))],
        )


class _ScriptedClient:
    def __init__(self, handler):
        self.chat = SimpleNamespace(completions=_ScriptedCompletions(handler))
        self.options_history = []

    def with_options(self, **kwargs):
        self.options_history.append(kwargs)
        return self


def _synthetic_fixture():
    text58 = (
        "Điều 58. Điều kiện và thể thức tiến hành họp Hội đồng thành viên\n"
        "1. Cuộc họp Hội đồng thành viên được tiến hành khi có số thành viên dự họp "
        "sở hữu từ 65% vốn điều lệ trở lên; tỷ lệ cụ thể do Điều lệ công ty quy định.\n"
    )
    text59 = (
        "Điều 59. Nghị quyết, quyết định của Hội đồng thành viên\n"
        "1. Hội đồng thành viên thông qua nghị quyết thuộc thẩm quyền bằng biểu quyết.\n"
    )
    chunks = [
        {
            "chunk_id": "luat_dn_2020:d58:p0", "doc_id": "luat_dn_2020", "ten_vb": "Luật Doanh nghiệp",
            "ten_ngan": "LDN", "so_hieu": "59/2020/QH14", "dieu": "58", "khoan": ["1"],
            "loai_hinh": "tnhh_2tv", "vai_tro": "chinh", "trang_thai": "con_hieu_luc", "hieu_luc": True,
            "ngay_hieu_luc": "2021-01-01", "sua_doi_cho": None, "sua_doi": [], "refs": [], "text": text58,
        },
        {
            "chunk_id": "luat_dn_2020:d59:p0", "doc_id": "luat_dn_2020", "ten_vb": "Luật Doanh nghiệp",
            "ten_ngan": "LDN", "so_hieu": "59/2020/QH14", "dieu": "59", "khoan": ["1"],
            "loai_hinh": "tnhh_2tv", "vai_tro": "chinh", "trang_thai": "con_hieu_luc", "hieu_luc": True,
            "ngay_hieu_luc": "2021-01-01", "sua_doi_cho": None, "sua_doi": [], "refs": [], "text": text59,
        },
    ]
    manifest = {
        "documents": [{
            "doc_id": "luat_dn_2020", "ten_vb": "Luật Doanh nghiệp", "ten_ngan": "Luật Doanh nghiệp",
            "so_hieu": "59/2020/QH14", "loai_vb": "luat", "vai_tro": "chinh",
            "trang_thai": "con_hieu_luc", "ngay_hieu_luc": "2021-01-01", "sua_doi_cho": None,
        }],
    }
    bm25, bm25_noacc = build_bm25_indices(chunks)
    emb = np.ones((len(chunks), 4), dtype=np.float32) / 2.0
    retriever = Retriever(
        chunks=chunks, emb=emb, embedder=_DummyEmbedder(),
        reranker=_DummyReranker(), bm25=bm25, bm25_noacc=bm25_noacc,
    )
    question = "Công ty TNHH hai thành viên hỏi điều kiện họp và nghị quyết theo Điều 58 và Điều 59 Luật Doanh nghiệp?"
    c_hash = corpus_semantic_hash(chunks)
    k1_start = text58.index("1. Cuộc họp")
    smoke = {
        "schema_version": 1,
        "annotation_status": "draft_requires_user_review",
        "corpus_sha256": c_hash,
        "cases": [{
            "case_id": "Q015",
            "origin": "existing_golden_copy",
            "source_golden_id": "Q015",
            "corpus_sha256": c_hash,
            "annotation_status": "draft_requires_user_review",
            "assessment_status": "needs_user_review",
            "question": question,
            "category": "theo_so_dieu",
            "expected_behavior": "answer",
            "expected_articles": ["luat_dn_2020:58", "luat_dn_2020:59"],
            "issues": [{
                "issue_id": "Q015:i1",
                "title": "Điều kiện họp",
                "assessment_status": "needs_user_review",
                "requirements": [{
                    "requirement_id": "Q015:r1",
                    "assessment_status": "needs_user_review",
                    "alternatives": [{
                        "group_id": "Q015:g1",
                        "spans": [{
                            "parent_chunk_id": "luat_dn_2020:d58:p0",
                            "doc_id": "luat_dn_2020",
                            "dieu": "58",
                            "khoan": "1",
                            "start": k1_start,
                            "end": len(text58),
                            "quote": text58[k1_start:],
                        }],
                    }],
                }],
            }],
        }],
    }
    golden = [
        {"id": "Q015", "question": question, "gold": ["luat_dn_2020:58", "luat_dn_2020:59"], "type": "theo_so_dieu", "should_refuse": False},
        {"id": "R001", "question": "Câu hỏi ngoài phạm vi từ chối?", "gold": [], "type": "refusal", "should_refuse": True},
    ]
    return chunks, manifest, retriever, smoke, golden, question


def test_import_does_not_load_pipeline_llm_or_model_libraries():
    code = (
        "import sys; sys.path.insert(0, " + repr(str(ROOT)) + "); "
        "sys.path.insert(0, " + repr(str(ROOT / "src")) + "); "
        "import scripts.check_query_pipeline; "
        "assert not any(name in sys.modules for name in ("
        "'ragdn.pipeline','ragdn.llm','ragdn.generate','sentence_transformers','torch','openai'))"
    )
    res = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize("argv", [
    [],
    ["--suite", "smoke", "--mode", "adaptive"],
    ["--suite", "smoke", "--retrieval-only"],
    ["--suite", "golden", "--retrieval-only"],
    ["--suite", "compare", "--mode", "adaptive", "--retrieval-only"],
    ["--suite", "smoke", "--mode", "adaptive", "--retrieval-only", "--with-answer"],
    ["--preflight", "--k", "0"],
    ["--preflight", "--suite", "smoke", "--mode", "adaptive", "--retrieval-only"],
    ["--preflight", "--with-answer"],
])
def test_cli_argument_gating_prevents_accidental_inference(argv):
    called = []
    rc = runner_main(
        argv,
        ps_fetcher=lambda _url: called.append("ps") or {"status": "reachable", "running_models": []},
        rag_factory=lambda: called.append("rag"),
    )
    assert rc == 2
    assert called == []


def test_preflight_checks_real_workspace_and_strips_credentials():
    env = {
        "LLM_BASE_URL": "http://admin:super_secret_token@localhost:11434/v1?api_key=leak",
        "LLM_MODEL": "qwen2.5:3b",
    }
    report = run_preflight(
        environ=env,
        ps_fetcher=lambda url: {
            "status": "reachable",
            "running_models": [{"name": "qwen2.5:3b", "model": "qwen2.5:3b", "context_length": 32768}],
        },
    )
    assert report["status"] == "ready"
    assert report["corpus"]["chunk_count"] == 676
    assert report["smoke_artifact"]["status"] == "valid"
    assert report["smoke_artifact"]["case_count"] == 24
    assert report["smoke_artifact"]["locator_count"] == 99
    assert report["runtime"]["confirmed_context_tokens"] == 32768
    assert report["runtime"]["context_source"] == "ollama_api_ps"
    serialized = json.dumps(report)
    assert "super_secret_token" not in serialized and "leak" not in serialized


def test_preflight_detects_missing_deps_index_hash_mismatch_and_unknown_context(tmp_path):
    dep_fail = run_preflight(
        environ={"RAG_QUERY_CONTEXT_TOKENS": "16384"},
        spec_finder=lambda name: None if name == "sentence_transformers" else True,
        probe_ollama=False,
    )
    assert dep_fail["status"] == "failed"
    assert "missing_dependencies:sentence_transformers" in dep_fail["errors"]

    no_ctx = run_preflight(
        environ={"LLM_MODEL": "qwen2.5:3b"},
        ps_fetcher=lambda _url: {"status": "unreachable", "running_models": []},
    )
    assert no_ctx["status"] == "failed"
    assert "context_budget_unconfirmed" in no_ctx["errors"]
    assert no_ctx["runtime"]["confirmed_context_tokens"] is None
    assert no_ctx["guidance"]

    bad_meta = tmp_path / "meta.json"
    bad_meta.write_text(json.dumps({"n": 676, "content_sha256": "stale", "model": "BAAI/bge-m3"}), encoding="utf-8")
    hash_fail = run_preflight(
        meta_path=bad_meta,
        environ={"RAG_QUERY_CONTEXT_TOKENS": "16384"},
        probe_ollama=False,
    )
    assert hash_fail["status"] == "failed"
    assert "index_content_hash_mismatch" in hash_fail["errors"]

    missing_idx = run_preflight(
        emb_path=tmp_path / "absent.npy",
        environ={"RAG_QUERY_CONTEXT_TOKENS": "16384"},
        probe_ollama=False,
    )
    assert missing_idx["status"] == "failed"
    assert "missing_index_files" in missing_idx["errors"]


def test_run_directory_collision_and_protected_file_refusal(tmp_path):
    out_root = tmp_path / "runs"
    first = create_run_directory(out_root, run_id="fixed_run")
    write_run_report(first, {"ok": True})
    with pytest.raises(FileExistsError):
        create_run_directory(out_root, run_id="fixed_run")
    with pytest.raises(FileExistsError):
        write_run_report(first, {"ok": False})
    with pytest.raises(ValueError, match="protected"):
        create_run_directory(ROOT / "eval" / "golden_check.jsonl")


def test_compare_abcd_shares_budgets_reuses_planner_and_keeps_legacy_vs_draft_metrics_distinct(capsys):
    chunks, manifest, retriever, smoke, golden, question = _synthetic_fixture()
    planner_calls = []

    def handler(kwargs, _call_no):
        user_payload = json.loads(kwargs["messages"][1]["content"])
        if "sources" not in user_payload:
            planner_calls.append(user_payload["question"])
            return json.dumps({
                "facts": [{"fact_id": "f1", "quote": question, "is_unknown": False}],
                "issues": [{
                    "issue_id": "i1", "title": "Điều kiện họp", "fact_ids": ["f1"],
                    "scope": {"subjects": [], "company_types": ["tnhh_2tv"], "doc_ids": ["luat_dn_2020"], "time_quote": None},
                    "main_query": "Điều 58 Luật Doanh nghiệp điều kiện họp Hội đồng thành viên",
                    "backup_query": "Điều 58 Luật Doanh nghiệp tỷ lệ dự họp",
                }],
                "reference_quotes": [],
                "unprocessed_parts": [],
            }, ensure_ascii=False)
        group = user_payload["sources"][0]
        return json.dumps({
            "evidence": [{
                "evidence_id": "e1",
                "issue_id": "i1",
                "group_id": group["group_id"],
                "unit_ids": [u["unit_id"] for u in group["units"]],
                "window_ids": list(group.get("window_ids", [])),
                "doc_id": "luat_dn_2020",
                "dieu": "58",
                "purpose": "current_law",
            }],
            "issues": [{
                "issue_id": "i1",
                "status": "supported",
                "evidence_ids": ["e1"],
                "missing_information": [],
            }],
        }, ensure_ascii=False)

    client = _ScriptedClient(handler)
    cfg = QueryConfig(mode="adaptive", context_tokens=32768)
    report = execute_compare_suite(
        smoke, chunks, retriever, golden_rows=golden, k=2, config=cfg,
        context_tokens=32768, manifest=manifest, client=client, model="qwen2.5:3b",
    )
    assert set(report["branches"]) == {"A", "B", "C", "D"}
    assert report["planner_reuse"]["enabled"] is True
    assert report["planner_reuse"]["recorded_from_c_count"] == 1
    assert report["planner_reuse"]["supplied_to_d_case_ids"] == ["Q015"]
    assert len(planner_calls) == 1
    for branch in ("A", "B", "C", "D"):
        b_rep = report["branches"][branch]
        assert b_rep["legacy_article_retrieval"]["basis"] == (
            "approved_existing_golden_article_labels_independent_of_draft_span_annotations"
        )
        assert b_rep["legacy_article_retrieval"]["buckets"]["ALL"]["status"] == "assessed"
        assert b_rep["smoke_annotation_evaluation"]["evidence_coverage"]["status"] == "not_assessed"
        assert b_rep["smoke_annotation_evaluation"]["acceptance"] == "user_review_required"
    assert report["branches"]["D"]["case_observations"]["Q015"]["completion_calls"] == 1
    assert report["branches"]["C"]["case_observations"]["Q015"]["completion_calls"] == 1
    b_obs = report["branches"]["B"]["case_observations"]["Q015"]
    c_obs = report["branches"]["C"]["case_observations"]["Q015"]
    d_obs = report["branches"]["D"]["case_observations"]["Q015"]
    assert b_obs["trace"]["packing_estimates_by_stage"]["mapper"] > 0
    assert b_obs["context_tokens"] == 0  # Packed candidate context was never sent.
    assert set(c_obs["trace"]["prompt_tokens_by_stage"]) == {"planner"}
    assert set(d_obs["trace"]["prompt_tokens_by_stage"]) == {"mapper"}
    assert c_obs["context_tokens"] == c_obs["trace"]["prompt_tokens_by_stage"]["planner"]
    assert all(options["max_retries"] == 0 for options in client.options_history)
    assert all("Q015:r1" not in json.dumps(call["messages"]) for call in client.chat.completions.calls)
    serialized = json.dumps(report, ensure_ascii=False)
    assert question not in serialized
    assert "65% vốn điều lệ" not in serialized
    output = capsys.readouterr()
    progress = [line for line in output.err.splitlines() if line.startswith("[progress]")]
    assert output.out == "" and len(progress) == 8
    for index, branch in enumerate(("A", "B", "C", "D")):
        before, after = progress[index * 2:index * 2 + 2]
        assert f"suite=compare mode={'legacy' if branch == 'A' else 'adaptive'} variant={branch}" in before
        assert 'case_id="Q015" 1/1 execution=started' in before
        assert f"variant={branch}" in after and "execution=completed" in after
    assert question not in output.err and "65% vốn điều lệ" not in output.err


def test_smoke_and_golden_execution_with_answer_operational_metrics_and_privacy():
    chunks, manifest, retriever, smoke, golden, question = _synthetic_fixture()
    bad_client = _ScriptedClient(lambda _kw, _n: "{not valid json")
    smoke_rep = execute_smoke_suite(
        smoke, chunks, retriever, golden_rows=golden, mode="adaptive",
        with_answer=True, k=2, context_tokens=32768, manifest=manifest,
        client=bad_client, model="qwen2.5:3b",
        answer_runner=lambda _q, *, query_result=None, mode="adaptive": {
            "answer": "Câu trả lời nhạy cảm không được lộ",
            "sources": list(query_result.sources),
            "refused": False,
            "partial": False,
        },
    )
    obs = smoke_rep["case_observations"]["Q015"]
    assert obs["fallback"] is True and obs["invalid_json"] is True
    assert obs["trace"]["answer_execution"]["executed"] is True
    assert obs["context_tokens"] is None  # Dummy reader supplied no actual prompt counts.
    assert smoke_rep["smoke_annotation_evaluation"]["answer_quality"]["false_supported"]["status"] == "not_assessed"

    gold_rep = execute_golden_suite(
        golden, chunks, retriever, mode="legacy", with_answer=False, k=2,
        context_tokens=32768, manifest=manifest,
    )
    assert gold_rep["legacy_article_retrieval"]["refusal_cases_excluded"] == 1
    assert gold_rep["legacy_article_retrieval"]["buckets"]["ALL"]["n"] == 1
    serialized = json.dumps({"smoke": smoke_rep, "golden": gold_rep}, ensure_ascii=False)
    assert question not in serialized
    assert "Câu trả lời nhạy cảm" not in serialized


def test_p5_rag_and_generate_answer_integration_via_default_runner(monkeypatch: pytest.MonkeyPatch):
    import ragdn.generate as gen_mod
    from ragdn.pipeline import RAG

    chunks, manifest, retriever, smoke, golden, _q = _synthetic_fixture()
    llm_calls = []

    def fake_call_llm(system: str, user: str, history=None) -> str:
        llm_calls.append((system, user, history))
        return "Cuộc họp Hội đồng thành viên được tiến hành khi dự họp sở hữu từ 65% vốn điều lệ trở lên [1]."

    monkeypatch.setattr(gen_mod, "call_llm", fake_call_llm)
    rag = RAG(manifest=manifest, retriever=retriever)

    ret_out = rag.retrieve(
        "Điều 58 Luật Doanh nghiệp quy định gì?",
        mode="adaptive",
        config=QueryConfig(mode="adaptive", context_tokens=32768),
    )
    assert ret_out["query_result"] is not None
    assert len(ret_out["context_spans"]) >= 1

    gold_rep = execute_golden_suite(
        golden[:1], chunks, rag.retriever, mode="legacy", with_answer=True, k=2,
        context_tokens=32768, manifest=manifest,
    )
    assert gold_rep["with_answer"] is True
    assert len(llm_calls) == 1
    assert gold_rep["case_observations"]["Q015"]["trace"]["answer_execution"]["executed"] is True
    observation = gold_rep["case_observations"]["Q015"]
    assert observation["trace"]["reader_context"]["reader_calls"] == 1
    actual_reader_bytes = len(llm_calls[0][0].encode("utf-8")) + len(llm_calls[0][1].encode("utf-8"))
    assert observation["context_tokens"] == actual_reader_bytes
    assert observation["trace"]["total_completion_calls"] == 1
    assert "Cuộc họp Hội đồng thành viên được tiến hành" not in json.dumps(gold_rep, ensure_ascii=False)


@pytest.mark.parametrize("run_id", ["../escape", "..\\escape", "/absolute", "C:\\escape", ".", "..", " "])
def test_run_ids_cannot_escape_exclusive_output_root(tmp_path, run_id):
    with pytest.raises(ValueError):
        create_run_directory(tmp_path, run_id=run_id)
    assert list(tmp_path.iterdir()) == []


def test_report_filename_cannot_escape_run_or_overwrite_protected_outputs(tmp_path):
    for filename in ("../report.json", "..\\report.json", str(ROOT / "eval/golden_check.jsonl")):
        with pytest.raises(ValueError):
            write_run_report(tmp_path, {}, filename)
    assert list(tmp_path.iterdir()) == []


def test_preflight_reports_missing_cache_without_model_import_or_download():
    report = run_preflight(environ={"RAG_QUERY_CONTEXT_TOKENS": "32768"},
                           probe_ollama=False, model_cache_checker=lambda _model, _env: False)
    assert report["status"] == "failed" and "local_model_cache_incomplete" in report["errors"]
    assert report["runtime"]["model_loading"] == "cache_only_no_implicit_download"
    assert report["fingerprints"]["smoke_annotation"]["sha256"]
    assert report["fingerprints"]["embeddings"]["bytes"] > 0


def test_model_cache_probe_handles_sharded_weights_read_only(tmp_path):
    snapshot = tmp_path / "hub/models--owner--model/snapshots/hash"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "part1.safetensors", "b": "part2.safetensors"}}), encoding="utf-8")
    env = {"HF_HOME": str(tmp_path)}
    assert not _cached_model_available("owner/model", env)
    (snapshot / "part1.safetensors").write_bytes(b"fixture-one")
    assert not _cached_model_available("owner/model", env)
    (snapshot / "part2.safetensors").write_bytes(b"fixture-two")
    assert _cached_model_available("owner/model", env)


def test_preflight_rejects_truncated_embedding_and_invalid_manifest(tmp_path):
    partial = tmp_path / "partial.npy"
    np.save(partial, np.zeros((676, 4), dtype=np.float32))
    partial.write_bytes(partial.read_bytes()[:-4])
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("[]", encoding="utf-8")
    result = run_preflight(emb_path=partial, manifest_path=manifest,
                          environ={"RAG_QUERY_CONTEXT_TOKENS": "32768"}, probe_ollama=False)
    assert "corrupt_index_files:ValueError" in result["errors"]
    assert "invalid_manifest_structure" in result["errors"]


def test_endpoint_credentials_and_invalid_ports_never_leak_or_trigger_probe():
    calls = []
    result = run_preflight(environ={"LLM_BASE_URL": "http://user:private@localhost:broken/v1",
                                   "RAG_QUERY_CONTEXT_TOKENS": "32768"},
                           ps_fetcher=lambda url: calls.append(url))
    assert "invalid_reader_endpoint" in result["errors"] and not calls
    assert "private" not in json.dumps(result)


def test_read_only_ollama_probe_drops_malformed_upstream_body():
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return None
        def read(self):
            return b"not-json private_upstream_body"
    result = fetch_ollama_ps("http://localhost:11434/api/ps", opener=lambda *_a, **_k: Response())
    assert result["status"] == "unreachable"
    assert "private_upstream_body" not in json.dumps(result)


def test_preflight_action_never_loads_rag_or_executes_pipeline(capsys):
    calls = []
    code = runner_main(["--preflight", "--no-ollama-ps"],
        environ={"RAG_QUERY_CONTEXT_TOKENS": "32768"},
        ps_fetcher=lambda url: calls.append(url), rag_factory=lambda: calls.append("load"),
        pipeline_runner=lambda *_a, **_k: calls.append("pipeline"))
    assert code == 0 and calls == []
    assert json.loads(capsys.readouterr().out)["status"] == "ready"


def _main_fixture(monkeypatch, tmp_path):
    chunks, manifest, retriever, smoke, golden, _ = _synthetic_fixture()
    smoke_path, gold_path = tmp_path / "smoke.json", tmp_path / "gold.jsonl"
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    gold_path.write_text("\n".join(json.dumps(row) for row in golden[:1]), encoding="utf-8")
    monkeypatch.setattr(runner, "SMOKE_PATH", smoke_path)
    monkeypatch.setattr(runner, "GOLDEN_PATH", gold_path)
    monkeypatch.setattr(runner, "run_preflight", lambda **_kwargs: {
        "status": "ready", "corpus": {"chunk_count": len(chunks), "corpus_sha256": corpus_semantic_hash(chunks)},
        "runtime": {"confirmed_context_tokens": 32768}})
    return SimpleNamespace(retriever=retriever, manifest=manifest)


def test_explicit_main_suite_writes_exclusive_report_and_loads_cache_only(monkeypatch, tmp_path, capsys):
    fake_rag = _main_fixture(monkeypatch, tmp_path)
    before = {name: runner.os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    def load():
        assert runner.os.environ["HF_HUB_OFFLINE"] == runner.os.environ["TRANSFORMERS_OFFLINE"] == "1"
        return fake_rag
    monkeypatch.setitem(sys.modules, "ragdn.pipeline", SimpleNamespace(RAG=SimpleNamespace(load=load)))
    output = tmp_path / "reports"
    rc = runner_main(["--suite", "golden", "--mode", "legacy", "--retrieval-only",
                      "--output-root", str(output), "--run-id", "first"], environ={})
    assert rc == 0
    result = json.loads((output / "first/report.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["evaluation"]["legacy_article_retrieval"]["buckets"]["ALL"]["status"] == "assessed"
    assert {name: runner.os.environ.get(name) for name in before} == before
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "completed"
    progress = [line for line in output.err.splitlines() if line.startswith("[progress]")]
    assert [line.split("stage=", 1)[1].split()[0] for line in progress] == [
        "preflight", "preflight", "model_load", "model_load", "case", "case"]


def test_expected_service_failure_keeps_content_free_failure_report(monkeypatch, tmp_path, capsys):
    fake_rag = _main_fixture(monkeypatch, tmp_path)
    def fail(*_args, **_kwargs):
        raise OSError("private question / upstream response must not leak")
    output = tmp_path / "reports"
    rc = runner_main(["--suite", "smoke", "--mode", "legacy", "--retrieval-only",
                      "--output-root", str(output), "--run-id", "failed"], environ={},
                     rag_factory=lambda: fake_rag, pipeline_runner=fail)
    assert rc == 1
    report = json.loads((output / "failed/report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["error_type"] == "OSError"
    output = capsys.readouterr()
    assert "private question" not in json.dumps(report) + output.out + output.err
    case_progress = [line for line in output.err.splitlines() if "stage=case" in line]
    assert len(case_progress) == 2
    assert "execution=started" in case_progress[0] and "execution=stopped" in case_progress[1]


@pytest.mark.parametrize("method,expected", [("conservative_utf8_bytes", 120), ("verified_reader_tokenizer", None)])
def test_reader_prompt_counts_are_consumed_without_mixing_accounting_methods(method, expected):
    chunks, manifest, retriever, _smoke, golden, _question = _synthetic_fixture()
    report = execute_golden_suite(golden[:1], chunks, retriever, mode="legacy", with_answer=True,
        context_tokens=32768, manifest=manifest, answer_runner=lambda *_a, **_kw: {
            "answer": "private", "sources": [], "processing": {"reader_context": {
                "method": method, "context_tokens": 32768, "prompt_tokens": {"reader": 100, "repair": 20},
                "reader_calls": 2}}})
    observation = report["case_observations"]["Q015"]
    assert observation["context_tokens"] == expected
    assert observation["trace"]["total_completion_calls"] == 2
    assert observation["trace"]["reader_context"]["prompt_tokens"]["repair"] == 20


def test_reusing_a_golden_id_with_a_changed_question_cannot_claim_approved_label_metrics():
    chunks, manifest, retriever, smoke, golden, _question = _synthetic_fixture()
    smoke["cases"][0]["question"] = "Điều 59 Luật Doanh nghiệp có nội dung gì?"
    report = execute_smoke_suite(smoke, chunks, retriever, golden_rows=golden, mode="legacy",
                                 context_tokens=32768, manifest=manifest)
    assert report["golden_label_question_mismatch_case_ids"] == ["Q015"]
    assert report["legacy_article_retrieval"]["buckets"]["ALL"]["status"] == "not_assessed"


def test_preflight_rejects_changed_verbatim_golden_question(tmp_path):
    artifact = json.loads(runner.SMOKE_PATH.read_text(encoding="utf-8"))
    artifact["cases"][0]["question"] = "Changed original question"
    target = tmp_path / "changed-smoke.json"
    target.write_text(json.dumps(artifact), encoding="utf-8")
    report = run_preflight(smoke_path=target, environ={"RAG_QUERY_CONTEXT_TOKENS": "32768"}, probe_ollama=False)
    assert "smoke_golden_question_mismatch" in report["errors"]


def test_legacy_rerank_pairs_count_actual_predict_calls_without_mutating_shared_retriever():
    chunks, manifest, retriever, _smoke, golden, _question = _synthetic_fixture()
    original_reranker = retriever.reranker
    report = execute_golden_suite(golden[:1], chunks, retriever, mode="legacy",
                                 context_tokens=32768, manifest=manifest)
    observation = report["case_observations"]["Q015"]
    assert observation["rerank_pairs"] == 2
    assert observation["trace"]["bounded_rerank_pairs"] == 0
    assert "rerank" in observation["stage_seconds"]
    assert retriever.reranker is original_reranker
    again = execute_golden_suite(golden[:1], chunks, retriever, mode="legacy",
                                context_tokens=32768, manifest=manifest)
    assert again["case_observations"]["Q015"]["rerank_pairs"] == 2


@pytest.mark.parametrize("suite", ["smoke", "golden"])
def test_case_progress_is_flushed_to_stderr_and_finishes_after_answer(monkeypatch, capsys, suite):
    chunks, manifest, retriever, smoke, golden, question = _synthetic_fixture()
    second = deepcopy(smoke["cases"][0])
    second["case_id"] = "S002"
    smoke["cases"].append(second)

    class ProgressStream(StringIO):
        def __init__(self):
            super().__init__()
            self.flushed = []
        def flush(self):
            self.flushed.append(self.getvalue())

    stream = ProgressStream()
    monkeypatch.setattr(runner.sys, "stderr", stream)
    answers = []
    def answer(_question, *, query_result, mode):
        # The case cannot be reported completed while generation is pending.
        assert "execution=started" in stream.getvalue().splitlines()[-1]
        answers.append(_question)
        return {"answer": "private_answer", "sources": list(query_result.sources)}

    options = dict(mode="legacy", with_answer=True, context_tokens=32768,
                   manifest=manifest, answer_runner=answer)
    if suite == "smoke":
        report = execute_smoke_suite(smoke, chunks, retriever, golden_rows=golden, **options)
        expected_ids = ["Q015", "S002"]
    else:
        report = execute_golden_suite(golden, chunks, retriever, **options)
        expected_ids = ["Q015", "R001"]
    lines = stream.getvalue().splitlines()
    assert len(lines) == 4 and len(answers) == 2
    assert len(report["case_observations"]) == 2
    for index, case_id in enumerate(expected_ids, 1):
        before, after = lines[(index - 1) * 2:index * 2]
        assert f'suite={suite} mode=legacy case_id="{case_id}" {index}/2' in before
        assert "execution=started elapsed=0.0s" in before
        assert "execution=completed elapsed=" in after
        assert all(any(snapshot.endswith(line + "\n") for snapshot in stream.flushed)
                   for line in (before, after))
    assert question not in stream.getvalue() and "65% vốn điều lệ" not in stream.getvalue()
    assert "private_answer" not in stream.getvalue() and "passed" not in stream.getvalue()
    assert capsys.readouterr().out == ""


def test_case_progress_escapes_control_characters_and_does_not_swallow_errors(capsys):
    with pytest.raises(OSError):
        with runner._progress_step(stage="case", suite="smoke", mode="adaptive",
                                   case_id="Q015\nspoof", index=1, total=24):
            raise OSError("private upstream body")
    output = capsys.readouterr()
    assert output.out == ""
    assert len(output.err.splitlines()) == 2
    assert 'case_id="Q015\\nspoof" 1/24' in output.err
    assert "execution=stopped" in output.err and "execution=completed" not in output.err
    assert "private upstream body" not in output.err


def main():
    return pytest.main([__file__, *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
