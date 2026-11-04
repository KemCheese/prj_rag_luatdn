# src/ragdn/index.py
import hashlib
import itertools
import json
import re
import unicodedata

from rank_bm25 import BM25Okapi
from underthesea import word_tokenize

RE_WORD = re.compile(r"\w+", re.UNICODE)


def chunks_sha256(chunks: list[dict]) -> str:
    """Băm ID và văn bản chunk theo thứ tự hàng của ma trận embedding."""
    records = [(chunk["chunk_id"], chunk["text"]) for chunk in chunks]
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s.replace("đ", "d").replace("Đ", "D"))

def tok(t: str) -> list[str]:
    return [w.lower() for w in word_tokenize(unicodedata.normalize("NFC", t)) if RE_WORD.search(w)]

def tok_noacc(t: str) -> list[str]:
    syl = RE_WORD.findall(strip_accents(t).lower())
    return syl + [f"{a}_{b}" for a, b in itertools.pairwise(syl)]


def build_bm25_indices(chunks: list[dict]):
    """Khởi tạo 2 chỉ mục BM25 (có dấu & không dấu) từ danh sách chunk."""
    bm25 = BM25Okapi([tok(c["text"]) for c in chunks])
    bm25_noacc = BM25Okapi([tok_noacc(c["text"]) for c in chunks])
    return bm25, bm25_noacc
