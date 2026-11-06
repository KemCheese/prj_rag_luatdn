# src/ragdn/pipeline.py
from copy import deepcopy
import json
from typing import Any

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
from ragdn.generate import copy_source, generate_answer, serialize_processing
from ragdn.index import build_bm25_indices, chunks_sha256
from ragdn.manifest import load_manifest
from ragdn.query_config import QueryConfig
from ragdn.query_models import QuestionPlan, SourceSpan
from ragdn.query_pipeline import ContextView, ReaderContext, run_query_pipeline
from ragdn.retrieval import Retriever


class RAG:
    def __init__(
        self,
        manifest: dict,
        retriever: Retriever,
        *,
        query_config: QueryConfig | None = None,
        reader_context: ReaderContext | None = None,
    ):
        self.manifest = manifest
        self.retriever = retriever
        self.query_config = query_config
        self.reader_context = reader_context

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
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))

        content_hash = meta.get("content_sha256")
        if content_hash is None:
            raise ValueError("Index dùng định dạng cũ, cần chạy lại scripts/build_index.py.")
        if meta["n"] != len(chunks):
            raise ValueError(
                f"Số chunk ({len(chunks)}) khác số vector trong metadata ({meta['n']}), "
                "cần chạy lại scripts/build_index.py."
            )
        if content_hash != chunks_sha256(chunks):
            raise ValueError("Dữ liệu chunk đã thay đổi, cần chạy lại scripts/build_index.py.")
        emb = np.load(EMB_PATH)
        if emb.shape[0] != len(chunks):
            raise ValueError(
                f"Số chunk ({len(chunks)}) khác số vector ({emb.shape[0]}), "
                "cần chạy lại scripts/build_index.py."
            )

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

    def _resolve_config_and_context(
        self,
        mode: str | None = None,
        config: QueryConfig | None = None,
        context: ReaderContext | None = None,
        variant: str | None = None,
    ) -> tuple[QueryConfig, ReaderContext, str]:
        cfg = config or self.query_config or QueryConfig.from_env()
        if mode is not None:
            if mode not in {"adaptive", "legacy"}:
                raise ValueError(f"Unsupported mode: {mode!r}")
            cfg = cfg.model_copy(update={"mode": mode})
        if variant is not None:
            if variant not in {"A", "B", "C", "D"}:
                raise ValueError(f"Unsupported variant: {variant!r}")
            if variant == "A":
                cfg = cfg.model_copy(update={"mode": "legacy"})
                internal_variant = "D"
            else:
                internal_variant = variant
        else:
            internal_variant = "D"
        ctx = context or self.reader_context or ReaderContext(cfg.context_tokens)
        return cfg, ctx, internal_variant

    def retrieve(
        self,
        question: str,
        *,
        mode: str | None = None,
        config: QueryConfig | None = None,
        context: ReaderContext | None = None,
        variant: str | None = None,
        cached_plan: QuestionPlan | None = None,
        client: Any = None,
        model: str | None = None,
    ) -> dict:
        cfg, ctx, internal_variant = self._resolve_config_and_context(mode, config, context, variant)
        if hasattr(self.retriever, "candidate_batch"):
            qres = run_query_pipeline(
                question,
                self.retriever,
                config=cfg,
                context=ctx,
                manifest=self.manifest,
                client=client,
                model=model,
                variant=internal_variant,
                cached_plan=cached_plan,
            )
            sources = [copy_source(s) for s in qres.sources]
            views = list(qres.context_views)
            proc = serialize_processing(
                qres.processing,
                evidence=qres.evidence,
                context_views=views,
                sources=sources,
            )
            titles = {issue.issue_id: issue.title for issue in [*qres.plan.issues, *qres.plan.not_processed]} if qres.plan else {}
            for assessment in proc.get("issues", []):
                if assessment["issue_id"] in titles:
                    assessment["title"] = titles[assessment["issue_id"]]
            return {
                **copy_source(qres.retrieval),
                "sources": sources,
                "context_views": views,
                "context_spans": proc.get("context_spans", []),
                "plan": qres.plan,
                "evidence": list(qres.evidence),
                "processing": proc,
                "partial": qres.partial,
                "partial_reasons": list(qres.partial_reasons),
                "diagnostics": dict(qres.diagnostics),
                "direct_complete": qres.direct_complete,
                "query_result": qres,
            }

        raw = dict(self.retriever.retrieve(question))
        chunks = getattr(self.retriever, "chunks", None)
        sources = [deepcopy(dict(chunks[i])) for i in raw.get("hits", [])] if chunks is not None else []
        views = [
            ContextView(
                str(p.get("chunk_id", idx)),
                idx,
                (SourceSpan(start=0, end=len(p.get("text", "")), quote=p.get("text", "")),),
                p.get("text", ""),
                True,
            )
            for idx, p in enumerate(sources, 1) if p.get("text") and p.get("chunk_id") is not None
        ] if sources else []
        proc = serialize_processing(None, context_views=views, sources=sources)
        return {
            **raw,
            "sources": sources,
            "context_views": views,
            "context_spans": proc.get("context_spans", []),
            "plan": None,
            "evidence": [],
            "processing": proc,
            "partial": False,
            "partial_reasons": [],
            "diagnostics": {},
            "direct_complete": not raw.get("low_confidence", False),
            "query_result": None,
        }

    def answer(
        self,
        question: str,
        *,
        mode: str | None = None,
        config: QueryConfig | None = None,
        context: ReaderContext | None = None,
        variant: str | None = None,
        cached_plan: QuestionPlan | None = None,
        client: Any = None,
        model: str | None = None,
    ) -> dict:
        cfg, ctx, _ = self._resolve_config_and_context(mode, config, context, variant)
        ret = self.retrieve(
            question,
            mode=mode,
            config=cfg,
            context=ctx,
            variant=variant,
            cached_plan=cached_plan,
            client=client,
            model=model,
        )
        answer = generate_answer(
            question,
            self.retriever,
            retrieval=ret,
            context_views=ret.get("context_views"),
            query_result=ret.get("query_result"),
            config=cfg,
            context=ctx,
            manifest=self.manifest,
        )
        answer["diagnostics"] = copy_source(ret.get("diagnostics", {}))
        return answer
