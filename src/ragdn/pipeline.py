# src/ragdn/pipeline.py
import hashlib
import json

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from ragdn.config import (
    CHUNKS_PATH,
    EMB_PATH,
    MANIFEST_PATH,
    META_PATH,
    MODEL_EMBEDDING,
    MODEL_RERANKER,
)
from ragdn.generate import generate_answer
from ragdn.index import build_bm25_indices
from ragdn.manifest import load_manifest
from ragdn.retrieval import Retriever


class RAG:
    def __init__(self, manifest: dict, retriever: Retriever):
        self.manifest = manifest
        self.retriever = retriever

    @classmethod
    def load(cls):
        # 1. Load manifest
        manifest_obj = load_manifest(MANIFEST_PATH)
        manifest_dict = manifest_obj.model_dump()

        # 2. Load chunks
        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(f"Chưa tìm thấy {CHUNKS_PATH}. Hãy chạy scripts/build_corpus.py trước.")
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]

        # 3. Load embeddings & meta
        if not EMB_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError("Chưa tìm thấy index vector. Hãy chạy scripts/build_index.py trước.")
        emb = np.load(EMB_PATH)
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))

        ids = [c["chunk_id"] for c in chunks]
        cur_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        assert meta["n"] == len(chunks), f"Số chunk ({len(chunks)}) khác số vector ({meta['n']})"
        assert meta["ids_sha256"] == cur_hash, "Dữ liệu chunk đã thay đổi, cần chạy lại scripts/build_index.py"

        # 4. Load Models
        embedder = SentenceTransformer(MODEL_EMBEDDING)
        reranker = CrossEncoder(MODEL_RERANKER, max_length=512)

        # 5. Build BM25
        bm25, bm25_noacc = build_bm25_indices(chunks)

        # 6. Khởi tạo Retriever
        retriever = Retriever(
            chunks=chunks,
            emb=emb,
            embedder=embedder,
            reranker=reranker,
            bm25=bm25,
            bm25_noacc=bm25_noacc,
        )

        return cls(manifest=manifest_dict, retriever=retriever)

    def retrieve(self, question: str):
        return self.retriever.retrieve(question)

    def answer(self, question: str):
        return generate_answer(question, self.retriever)
