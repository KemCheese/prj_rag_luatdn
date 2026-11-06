from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import json
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    InternalServerError,
    NotFoundError,
)

import api.main as api
from ragdn.generate import DISCLAIMER, REFUSAL, UNVERIFIED

SOURCE = {
    "chunk_id": "test:d4:p0", "ten_vb": "Luật Doanh nghiệp", "dieu": "4",
    "khoan": ["1"], "page": 1, "hieu_luc": True, "sua_doi": [], "text": "Nguồn <gốc>",
}
UPSTREAM = httpx.Request("POST", "http://llm.invalid/v1/chat/completions")
PRIVATE_ERROR = "upstream-private-detail"
SERVICE_ERROR = "Dịch vụ LLM đang lỗi hoặc chưa sẵn sàng, vui lòng thử lại sau."


@pytest.fixture
def client(monkeypatch):
    rag = SimpleNamespace(
        manifest={"corpus_version": "test-version", "documents": []},
        answer=Mock(return_value={
            "answer": "Câu trả lời [1].", "refused": False,
            "disclaimer": DISCLAIMER, "sources": [SOURCE],
        }),
    )
    monkeypatch.setattr(api.RAG, "load", lambda: rag)
    api.limiter.reset()
    try:
        with TestClient(api.app, raise_server_exceptions=False) as http:
            yield http, rag
    finally:
        api.limiter.reset()


@pytest.mark.parametrize("error, message", [
    (NotFoundError(PRIVATE_ERROR, response=httpx.Response(404, request=UPSTREAM), body=None), SERVICE_ERROR),
    (InternalServerError(PRIVATE_ERROR, response=httpx.Response(500, request=UPSTREAM), body=None), SERVICE_ERROR),
    (APIError(PRIVATE_ERROR, request=UPSTREAM, body=None), SERVICE_ERROR),
    (TimeoutError(PRIVATE_ERROR), "Dịch vụ LLM quá thời gian, vui lòng thử lại."),
    (APITimeoutError(request=UPSTREAM), "Dịch vụ LLM quá thời gian, vui lòng thử lại."),
    (APIConnectionError(message=PRIVATE_ERROR, request=UPSTREAM), "Không thể kết nối tới dịch vụ LLM, vui lòng kiểm tra lại."),
])
def test_chat_llm_errors_return_safe_503(client, error, message):
    http, rag = client
    rag.answer.side_effect = error

    response = http.post("/chat", json={"question": "Câu hỏi hợp lệ?"})

    assert response.status_code == 503
    assert response.json() == {"detail": message}
    assert PRIVATE_ERROR not in response.text
    assert "llm.invalid" not in response.text


def test_chat_logs_llm_error_without_private_body(client, caplog):
    http, rag = client
    rag.answer.side_effect = APIError(PRIVATE_ERROR, request=UPSTREAM, body=None)

    http.post("/chat", json={"question": "Câu hỏi hợp lệ?"})

    assert any(record.name == "api.main" for record in caplog.records)
    assert PRIVATE_ERROR not in caplog.text


def test_chat_does_not_disguise_programming_errors_as_llm_failures(client):
    http, rag = client
    rag.answer.side_effect = ValueError("internal programming bug")

    response = http.post("/chat", json={"question": "Câu hỏi hợp lệ?"})

    assert response.status_code == 500
    assert "internal programming bug" not in response.text


@pytest.mark.parametrize("answer, refused", [
    ("Câu trả lời [1].", False), (REFUSAL, True), (UNVERIFIED, False),
])
def test_chat_keeps_answer_disclaimer_and_sources_separate(client, answer, refused):
    http, rag = client
    rag.answer.return_value.update(answer=answer, refused=refused)

    response = http.post("/chat", json={"question": "Câu hỏi hợp lệ?"})

    assert response.status_code == 200
    body = response.json()
    assert {key: body[key] for key in ("answer", "refused", "disclaimer", "sources", "corpus_version")} == {
        "answer": answer, "refused": refused, "disclaimer": DISCLAIMER,
        "sources": [{**SOURCE, "doc_id": None, "evidence_spans": []}], "corpus_version": "test-version",
    }
    assert body["partial"] is False and body["partial_reasons"] == []
    assert body["processing"]["status"] == "not_assessed"
    assert body["processing"]["mode"] == "legacy"


@pytest.mark.parametrize("question", ["ab", "x" * 2001])
def test_chat_validation_rejects_invalid_lengths_before_inference(client, question):
    http, rag = client

    response = http.post("/chat", json={"question": question})

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    rag.answer.assert_not_called()


def test_chat_rate_limit_still_returns_slowapi_error(client):
    http, rag = client
    responses = [http.post("/chat", json={"question": "Câu hỏi hợp lệ?"}) for _ in range(11)]

    assert all(response.status_code == 200 for response in responses[:10])
    assert responses[-1].status_code == 429
    assert "error" in responses[-1].json()
    assert rag.answer.call_count == 10


def test_startup_and_readonly_routes_work_with_loaded_rag(client):
    http, _ = client

    assert http.get("/health").json() == {"status": "ok", "corpus_version": "test-version"}
    assert http.get("/documents").json() == []
    assert "<script>" in http.get("/").text


@pytest.fixture(scope="module")
def run_web_chat():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the embedded Web UI JavaScript")
    script = api.index().split("<script>", 1)[1].split("</script>", 1)[0]
    harness = Path(__file__).with_name("web_chat_harness.cjs")

    def run(**case):
        result = subprocess.run(
            [node, str(harness)], input=json.dumps({"script": script, **case}),
            capture_output=True, text=True, encoding="utf-8", check=True, timeout=10,
        )
        return json.loads(result.stdout)

    return run


@pytest.mark.parametrize("case, message", [
    ({"status": 503, "data": {"detail": "Dịch vụ LLM chưa sẵn sàng."}}, "Dịch vụ LLM chưa sẵn sàng."),
    ({"status": 429, "data": {"error": "Rate limit exceeded"}}, "quá nhiều"),
    ({"status": 422, "data": {"detail": [{"msg": "too short"}]}}, "3 đến 2000"),
    ({"status": 500, "invalidJson": True}, "không thể xử lý"),
    ({"status": 503, "data": None}, "không thể xử lý"),
    ({"status": 503, "data": {"detail": {"internal": "private"}}}, "không thể xử lý"),
])
def test_web_http_errors_are_visible_and_controls_recover(run_web_chat, case, message):
    result = run_web_chat(**case)

    assert message in result["text"]
    assert result["html"] == ""
    assert result["inputEnabled"] and result["buttonEnabled"] and result["focused"]
    assert "SyntaxError" not in result["text"]


def test_web_network_error_hides_raw_exception_and_restores_controls(run_web_chat):
    result = run_web_chat(networkError=PRIVATE_ERROR)

    assert "kết nối" in result["text"]
    assert PRIVATE_ERROR not in result["text"]
    assert result["inputEnabled"] and result["buttonEnabled"] and result["focused"]


def test_web_error_detail_is_rendered_as_text(run_web_chat):
    detail = '<img src=x onerror="alert(1)">'
    result = run_web_chat(status=503, data={"detail": detail})

    assert result["text"] == detail
    assert result["html"] == ""


@pytest.mark.parametrize("answer", ["Câu trả lời [1].", REFUSAL, UNVERIFIED])
def test_web_disclaimer_precedes_every_answer_once_and_preserves_sources(run_web_chat, answer):
    result = run_web_chat(data={"answer": answer, "disclaimer": DISCLAIMER, "sources": [SOURCE]})

    assert result["html"].count(DISCLAIMER) == 1
    assert result["html"].index(DISCLAIMER) < result["html"].index(answer.split(" [1]")[0])
    assert "Nguồn &lt;gốc&gt;" in result["html"]
    assert 'id="src-item-msg-1-1"' in result["html"]
    assert result["inputEnabled"] and result["buttonEnabled"]


def test_web_disclaimer_is_escaped(run_web_chat):
    result = run_web_chat(data={"answer": "Câu trả lời.", "disclaimer": "<b>Lưu ý</b>", "sources": []})

    assert "&lt;b&gt;Lưu ý&lt;/b&gt;" in result["html"]
    assert "<b>Lưu ý</b>" not in result["html"]


@pytest.mark.parametrize("data", [None, {}, {"answer": None}])
def test_web_malformed_success_is_not_rendered_as_empty_answer(run_web_chat, data):
    result = run_web_chat(data=data)

    assert "không hợp lệ" in result["text"]
    assert result["inputEnabled"] and result["buttonEnabled"]


@pytest.mark.parametrize("length", [501, 2000])
def test_chat_accepts_expanded_question_length_without_truncation(client, length):
    http, rag = client
    question = "x" * length
    assert http.post("/chat", json={"question": question}).status_code == 200
    rag.answer.assert_called_once_with(question)


def test_chat_preserves_processing_partial_and_exact_parent_evidence_spans(client):
    http, rag = client
    span = {"parent_chunk_id": SOURCE["chunk_id"], "doc_id": "d", "dieu": "4", "start": 0, "end": 5, "quote": "Nguồn"}
    processing = {"mode": "enhanced", "status": "partial", "issues": [{"issue_id": "i1", "status": "conditional"}],
                  "missing_information": [{"description": "Cần xác định Điều lệ"}],
                  "warnings": [], "context_spans": [span], "partial_reasons": ["information_missing"]}
    rag.answer.return_value.update(partial=True, partial_reasons=["information_missing"], processing=processing)
    body = http.post("/chat", json={"question": "Câu hỏi hợp lệ?", "variant": "B", "mode": "legacy"}).json()
    assert body["processing"] == processing
    assert body["partial"] is True and body["partial_reasons"] == ["information_missing"]
    assert body["sources"][0]["text"] == SOURCE["text"]
    assert body["sources"][0]["evidence_spans"] == [{key: span[key] for key in ("start", "end", "quote")}]
    rag.answer.assert_called_once_with("Câu hỏi hợp lệ?")


def test_web_processing_and_every_source_metadata_value_are_escaped(run_web_chat):
    payload = '<img src=x onerror="alert(1)">'
    data = {"answer": "Nội dung [1].", "disclaimer": DISCLAIMER, "partial": True,
            "processing": {"status": "partial", "issues": [{"issue_id": payload, "title": payload, "status": "conditional"}],
                           "missing_information": [{"description": payload}], "warnings": [payload]},
            "sources": [{**SOURCE, "ten_vb": payload, "page": payload, "khoan": [payload]}]}
    result = run_web_chat(data=data)
    assert 'class="processing-box"' in result["html"]
    assert "một phần" in result["html"] and "có điều kiện" in result["html"]
    assert payload not in result["html"]
    assert "&lt;img" in result["html"]
    assert result["inputEnabled"] and result["buttonEnabled"]


def test_web_generation_pruning_remains_visible_when_coverage_was_not_assessed(run_web_chat):
    result = run_web_chat(data={"answer": "Nội dung còn giữ lại.", "partial": True,
                               "partial_reasons": ["generation_pruned"],
                               "processing": {"status": "not_assessed", "mode": "baseline_fallback"}, "sources": []})
    assert "một phần" in result["html"] and "lược bỏ" in result["html"]
    assert "chưa được đánh giá" in result["html"]


def test_api_test_entrypoint_forwards_cli(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", [__file__, "-q", "-p", "no:cacheprovider"])
    monkeypatch.setattr(pytest, "main", lambda args: calls.append(args) or 7)
    assert main() == 7
    assert calls == [[__file__, "-q", "-p", "no:cacheprovider"]]


def main() -> int:
    return int(pytest.main([__file__, *sys.argv[1:]]))


if __name__ == "__main__":
    raise SystemExit(main())
