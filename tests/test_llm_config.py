# tests/test_llm_config.py
import runpy
from types import SimpleNamespace

import dotenv
import pytest

import ragdn.llm as llm_mod


def _load_llm(monkeypatch, **env):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    for name in ("LLM_TEMPERATURE", "LLM_TOP_P", "LLM_MAX_TOKENS", "LLM_SEED"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    # Namespace riêng giữ cấu hình của các test khác độc lập với test này.
    return SimpleNamespace(**runpy.run_path(llm_mod.__file__))


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
    llm = _load_llm(monkeypatch)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_response("Câu trả lời hoàn chỉnh [1].")

    monkeypatch.setattr(llm._client.chat.completions, "create", fake_create)
    history = [
        {"role": "user", "content": "<context>...</context>\n\n<question>Hỏi?</question>"},
        {"role": "assistant", "content": "Bản nháp 1 có lỗi [9]."},
    ]
    out = llm.call_llm("SYS", "Hãy sửa lỗi.", history=history)

    assert out == "Câu trả lời hoàn chỉnh [1]."
    assert captured["messages"] == [
        {"role": "system", "content": "SYS"},
        *history,
        {"role": "user", "content": "Hãy sửa lỗi."},
    ]
    assert captured["temperature"] == 0.05
    assert captured["top_p"] == 0.9
    assert captured["max_tokens"] == 2000
    assert captured["seed"] == 42


@pytest.mark.parametrize(("seed", "expected"), [("73", 73), ("0", 0), ("", None)])
def test_call_llm_passes_seed_when_configured(monkeypatch, seed, expected):
    llm = _load_llm(monkeypatch, LLM_SEED=seed)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_response("OK [1].")

    monkeypatch.setattr(llm._client.chat.completions, "create", fake_create)
    llm.call_llm("SYS", "USER")

    if expected is None:
        assert "seed" not in captured
    else:
        assert captured["seed"] == expected


def test_call_llm_respects_sampling_overrides(monkeypatch):
    llm = _load_llm(monkeypatch, LLM_TEMPERATURE="0", LLM_TOP_P="1", LLM_MAX_TOKENS="512")
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_response("OK [1].")

    monkeypatch.setattr(llm._client.chat.completions, "create", fake_create)
    llm.call_llm("SYS", "USER")

    assert captured["temperature"] == 0.0
    assert captured["top_p"] == 1.0
    assert captured["max_tokens"] == 512


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

