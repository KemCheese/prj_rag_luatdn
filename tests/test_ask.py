from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import APIError, APITimeoutError

from ragdn.generate import DISCLAIMER, REFUSAL, UNVERIFIED
from scripts import ask


@pytest.fixture
def rag(monkeypatch):
    rag = SimpleNamespace(answer=Mock())
    monkeypatch.setattr(ask.RAG, "load", lambda: rag)
    monkeypatch.setattr(ask, "load_dotenv", lambda: None)
    monkeypatch.setattr(ask.sys, "argv", ["scripts/ask.py", "Câu hỏi hợp lệ?"])
    return rag


@pytest.mark.parametrize("answer", ["Câu trả lời.", REFUSAL, UNVERIFIED])
def test_cli_prints_disclaimer_once_before_answer_and_keeps_sources(rag, capsys, answer):
    rag.answer.return_value = {
        "answer": answer, "disclaimer": DISCLAIMER,
        "sources": [{"chunk_id": "test:d4:p0", "ten_vb": "Luật Doanh nghiệp", "dieu": "4"}],
    }

    ask.main()

    output = capsys.readouterr()
    assert output.out.count(DISCLAIMER) == 1
    assert output.out.index(DISCLAIMER) < output.out.index(answer)
    assert "test:d4:p0 | Luật Doanh nghiệp, Điều 4" in output.out
    assert output.err == ""
    rag.answer.assert_called_once_with("Câu hỏi hợp lệ?")


@pytest.mark.parametrize("error", [
    APIError("upstream-private-detail", request=httpx.Request("POST", "http://llm.invalid"), body=None),
    APITimeoutError(request=httpx.Request("POST", "http://llm.invalid")),
])
def test_cli_llm_failure_exits_with_safe_stderr(rag, capsys, error):
    rag.answer.side_effect = error

    with pytest.raises(SystemExit) as exc:
        ask.main()

    output = capsys.readouterr()
    assert exc.value.code == 1
    assert "LLM" in output.err
    assert "upstream-private-detail" not in output.err
    assert "Traceback" not in output.err
    assert "CÂU TRẢ LỜI:" not in output.out


def test_cli_does_not_hide_programming_errors(rag):
    rag.answer.side_effect = ValueError("programming bug")

    with pytest.raises(ValueError, match="programming bug"):
        ask.main()


def test_cli_missing_question_keeps_usage_error(rag, monkeypatch, capsys):
    monkeypatch.setattr(ask.sys, "argv", ["scripts/ask.py"])

    with pytest.raises(SystemExit) as exc:
        ask.main()

    assert exc.value.code == 1
    assert "Cách dùng:" in capsys.readouterr().out
    rag.answer.assert_not_called()


def test_cli_mode_processing_and_debug_stay_after_sources_and_hide_payloads(rag, monkeypatch, capsys):
    monkeypatch.setattr(ask.sys, "argv", ["scripts/ask.py", "private-question", "--mode", "adaptive", "--debug-retrieval"])
    rag.answer.return_value = {"answer": "Câu trả lời.", "disclaimer": DISCLAIMER, "sources": [], "partial": True,
                               "processing": {"mode": "enhanced", "status": "partial", "budget": {"completion_calls": 2},
                                              "missing_information": [{"description": "Thiếu Điều lệ"}]},
                               "diagnostics": {"variant": "D", "prompt_tokens": {"planner": 42},
                                               "prompt": "private-prompt", "question": "private-question"}}
    ask.main()
    output = capsys.readouterr()
    rag.answer.assert_called_once_with("private-question", mode="adaptive")
    assert output.out.index("NGUỒN TRÍCH DẪN:") < output.out.index("PHẠM VI TRẢ LỜI:") < output.out.index("CHẨN ĐOÁN TRUY XUẤT:")
    assert "Thiếu Điều lệ" in output.out and '"completion_calls": 2' in output.out
    assert "private-question" not in output.out and "private-prompt" not in output.out
    assert output.err == ""


@pytest.mark.parametrize("mode_args", [[], ["--mode", "adaptive"]])
def test_cli_oversized_adaptive_falls_back_without_truncation(rag, monkeypatch, capsys, mode_args):
    question = "x" * 2001
    monkeypatch.setattr(ask.sys, "argv", ["scripts/ask.py", question, *mode_args])
    rag.answer.return_value = {"answer": "Câu trả lời.", "disclaimer": DISCLAIMER, "sources": []}
    ask.main()
    output = capsys.readouterr()
    rag.answer.assert_called_once_with(question, mode="legacy")
    assert "vượt 2000" in output.err
    assert "baseline_fallback" in output.out and "not_assessed" in output.out
    assert question not in output.out


def test_cli_explicit_legacy_accepts_long_question_without_adaptive_warning(rag, monkeypatch, capsys):
    question = "x" * 2001
    monkeypatch.setattr(ask.sys, "argv", ["scripts/ask.py", question, "--mode", "legacy"])
    rag.answer.return_value = {"answer": "Câu trả lời.", "disclaimer": DISCLAIMER, "sources": []}
    ask.main()
    assert capsys.readouterr().err == ""
    rag.answer.assert_called_once_with(question, mode="legacy")


def test_cli_test_entrypoint_forwards_cli(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", [__file__, "-q"])
    monkeypatch.setattr(pytest, "main", lambda args: calls.append(args) or 7)
    assert main() == 7
    assert calls == [[__file__, "-q"]]


def test_cli_technical_failure_labels_sources_as_related_not_verified(rag, capsys):
    rag.answer.return_value = {"answer": "Lượt xử lý chưa hoàn tất.", "disclaimer": DISCLAIMER,
                               "sources": [{"chunk_id": "s1", "ten_vb": "Luật Doanh nghiệp", "dieu": "58"}],
                               "processing": {"mode": "baseline_fallback", "status": "not_assessed", "source_use": "related_not_verified"}}
    ask.main()
    output = capsys.readouterr().out
    assert "NGUỒN LIÊN QUAN (CHƯA XÁC MINH LÀ CĂN CỨ TRẢ LỜI):" in output
    assert "NGUỒN TRÍCH DẪN:" not in output


def main() -> int:
    return int(pytest.main([__file__, *sys.argv[1:]]))


if __name__ == "__main__":
    raise SystemExit(main())
