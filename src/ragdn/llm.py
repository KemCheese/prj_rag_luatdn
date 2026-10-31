# src/ragdn/llm.py
import os
from dotenv import load_dotenv
from openai import OpenAI
import httpx

# Đảm bảo biến môi trường từ .env luôn được nạp trước khi khởi tạo client
load_dotenv()

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "200.0"))

_client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("LLM_API_KEY", "ollama"),
    timeout=httpx.Timeout(
        timeout=LLM_TIMEOUT,
        connect=10.0,  
        read=LLM_TIMEOUT, 
    ),
    max_retries=1,     
)

def call_llm(system: str, user: str, model: str | None = None) -> str:
    r = _client.chat.completions.create(
        model=model or os.environ.get("LLM_MODEL", "qwen2.5:3b"),
        temperature=0.3,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return r.choices[0].message.content or ""
