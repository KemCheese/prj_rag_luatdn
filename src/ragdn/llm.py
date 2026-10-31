# src/ragdn/llm.py
import logging
import os
import re

import httpx
from dotenv import load_dotenv
from openai import OpenAI

# Đảm bảo biến môi trường từ .env luôn được nạp trước khi khởi tạo client
load_dotenv()
log = logging.getLogger(__name__)

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "200"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.15"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1536"))
LLM_SEED = os.getenv("LLM_SEED") 

_client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("LLM_API_KEY", "ollama"),
    timeout=httpx.Timeout(timeout=LLM_TIMEOUT, connect=10.0, read=LLM_TIMEOUT),
    max_retries=LLM_MAX_RETRIES,
)

_RE_UPTO_LAST_END = re.compile(r"[\s\S]*[.;!?:\n](?:\s*\[\d+\])*")


def call_llm(system: str, user: str, model: str | None = None, *, history: list[dict] | None = None) -> str:
    r = _client.chat.completions.create(
        model=model or os.environ.get("LLM_MODEL", "qwen2.5:3b"),
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        max_tokens=LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            *(history or []),
            {"role": "user", "content": user},
        ],
        **({"seed": int(LLM_SEED)} if LLM_SEED else {}),
    )
    choice = r.choices[0]
    text = choice.message.content or ""
    if choice.finish_reason == "length":
        log.warning(
            "LLM chạm max_tokens=%d (prompt_tokens=%s) → bỏ câu dở dang cuối",
            LLM_MAX_TOKENS,
            getattr(r.usage, "prompt_tokens", "?"),
        )
        if m := _RE_UPTO_LAST_END.match(text):
            text = m.group(0)
    return text
