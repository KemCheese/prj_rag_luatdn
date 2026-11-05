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
