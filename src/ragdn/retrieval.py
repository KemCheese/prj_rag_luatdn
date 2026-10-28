# src/ragdn/retrieval.py
import re
import unicodedata

import numpy as np
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from ragdn.config import ALIASES_PATH, GLOSSARY_PATH
from ragdn.index import strip_accents, tok, tok_noacc

RE_Q_DIEU = re.compile(r"\bdieu\s+(\d+[a-z]?)\b")

class Retriever:
    def __init__(
        self,
        chunks: list[dict],
        emb: np.ndarray,
        embedder: SentenceTransformer,
        reranker: CrossEncoder,
        bm25: BM25Okapi,
        bm25_noacc: BM25Okapi,
        rerank_min: float = 0.15,
    ):
        self.chunks = chunks
        self.emb = emb
        self.embedder = embedder
        self.reranker = reranker
        self.bm25 = bm25
        self.bm25_noacc = bm25_noacc
        self.rerank_min = rerank_min

        self.valid = np.array([c["hieu_luc"] and c["vai_tro"] == "chinh" for c in chunks])
        self.by_article = {}
        for i, c in enumerate(chunks):
            self.by_article.setdefault((c["doc_id"], c["dieu"]), []).append(i)

        # Load glossary & aliases nếu có
        self.glossary = {}
        if GLOSSARY_PATH.exists():
            with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
                self.glossary = yaml.safe_load(f) or {}

        self.doc_aliases = []
        if ALIASES_PATH.exists():
            with open(ALIASES_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self.doc_aliases = [(item["pattern"], item["doc_id"]) for item in data.get("aliases", [])]

    def norm_query(self, q: str) -> str:
        q = unicodedata.normalize("NFC", q).strip()
        for k, v in self.glossary.items():
            q = re.sub(rf"(?i)\b{re.escape(k)}\b", f"{k} ({v})", q)
        return q

    def article_lookup(self, q: str):
        qa = strip_accents(q).lower()
        m = RE_Q_DIEU.search(qa)
        if not m:
            return [], None
        doc = next((d for pat, d in self.doc_aliases if re.search(pat, qa)), None)
        hits = [i for (d, dieu), idx in self.by_article.items()
                if dieu == m.group(1) and (doc is None or d == doc) for i in idx]
        ambiguous = doc is None and len({self.chunks[i]["doc_id"] for i in hits}) > 1
        return hits, ("ambiguous" if ambiguous else None)

    def topk(self, scores: np.ndarray, n: int):
        scores = np.where(self.valid, scores, -np.inf)
        return [int(i) for i in np.argsort(-scores)[:n] if np.isfinite(scores[i])]

    @staticmethod
    def rrf(rank_lists: list[list[int]], k: int = 60):
        s = {}
        for r in rank_lists:
            for pos, i in enumerate(r):
                s[i] = s.get(i, 0.0) + 1.0 / (k + pos + 1)
        return sorted(s, key=s.get, reverse=True)

    def retrieve(self, q: str, n_cand: int = 20, n_rerank: int = 8, top: int = 5):
        qn = self.norm_query(q)
        qv = self.embedder.encode(qn, normalize_embeddings=True)
        cand = self.rrf([
            self.topk(self.emb @ qv, n_cand),
            self.topk(self.bm25.get_scores(tok(qn)), n_cand),
            self.topk(self.bm25_noacc.get_scores(tok_noacc(qn)), n_cand),
        ])[:n_rerank]

        sc = self.reranker.predict([(qn, self.chunks[i]["text"]) for i in cand]) if cand else np.array([])
        ranked = [(cand[j], float(sc[j])) for j in np.argsort(-sc)[:top]]
        direct, flag = self.article_lookup(qn)
        hits = list(dict.fromkeys([i for i in direct[:4]] + [i for i, s in ranked if s >= self.rerank_min]))

        return {
            "hits": hits,
            "max_score": max((s for _, s in ranked), default=0.0),
            "low_confidence": not direct and (not ranked or ranked[0][1] < self.rerank_min),
            "flag": flag,
        }
