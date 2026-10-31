# tests/test_llm_config.py
from types import SimpleNamespace

import ragdn.llm as llm_mod


def _fake_response(content: str, finish_reason: str = "stop", prompt_tokens: int = 123):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens),
    )


def test_call_llm_passes_history_and_Sampling_params(monkeypatch):
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_response("Câu trả lời hoàn chỉnh [1].")

    monkeypatch.setattr(llm_mod._client.chat.completions, "create", fake_create)
    history = [
        {"role": "user", "content": "<context>...</context>\n\n<question>Hỏi?</question>"},
        {"role": "assistant", "content": "Bản nháp 1 có lỗi [9]."},
    ]
    out = llm_mod.call_llm("SYS", "Hãy sửa lỗi.", history=history)

    assert out == "Câu trả lời hoàn chỉnh [1]."
    assert captured["messages"] == [
        {"role": "system", "content": "SYS"},
        *history,
        {"role": "user", "content": "Hãy sửa lỗi."},
    ]
    assert captured["temperature"] == llm_mod.LLM_TEMPERATURE == 0.2
    assert captured["top_p"] == llm_mod.LLM_TOP_P == 0.9
    assert captured["max_tokens"] == llm_mod.LLM_MAX_TOKENS == 1536
    assert "seed" not in captured


def test_call_llm_passes_seed_when_configured(monkeypatch):
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_response("OK [1].")

    monkeypatch.setattr(llm_mod._client.chat.completions, "create", fake_create)
    monkeypatch.setattr(llm_mod, "LLM_SEED", "42", raising=False)
    llm_mod.call_llm("SYS", "USER")

    assert captured.get("seed") == 42


def test_call_llm_trims_dangling_sentence_on_length_finish(monkeypatch):
    raw = "Câu 1 đầy đủ [1]. Câu 2 cũng đầy đủ. [2] Câu 3 đang viết dở giữa chừng chưa có dấu chấm"
    monkeypatch.setattr(
        llm_mod._client.chat.completions,
        "create",
        lambda **kwargs: _fake_response(raw, finish_reason="length"),
    )

    out = llm_mod.call_llm("SYS", "USER")
    assert out == "Câu 1 đầy đủ [1]. Câu 2 cũng đầy đủ. [2]"


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))

