# src/ragdn/retrieval.py
import re
import unicodedata
from collections import OrderedDict, deque
from dataclasses import dataclass

import numpy as np
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from ragdn.article_references import ArticleReference, resolve_article_references
from ragdn.config import ALIASES_PATH, GLOSSARY_PATH
from ragdn.index import strip_accents, tok, tok_noacc
from ragdn.loai_hinh import detect_loai_hinh, is_conflict
from ragdn.source_windows import (
    SourceWindow, UnsupportedTokenizerError, WindowGroup, build_source_windows,
    encoded, pair_token_count, prepare_query, source_groups,
)

RE_Q_DIEU = re.compile(r"\b(?:điều|dieu)\s+(\d+(?:[a-zđ]|dd)?)\b")


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    chunk_index: int
    query_id: str
    issue_id: str
    query_text: str
    rrf_rank: int | None
    rrf_score: float | None
    channels: tuple[str, ...]
    direct: bool
    targeted: bool
    references: tuple[ArticleReference, ...] = ()


@dataclass(frozen=True)
class CandidateBatch:
    query_id: str
    issue_id: str
    candidates: tuple[Candidate, ...]
    unseen_parent_ids: tuple[str, ...]
    channel_available: dict[str, int]
    channel_unseen: dict[str, int]
    references: tuple[ArticleReference, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PooledParent:
    chunk_id: str
    chunk_index: int
    nominations: tuple[Candidate, ...]


@dataclass(frozen=True)
class CandidatePool:
    parents: tuple[PooledParent, ...]
    proposed_parent_ids: tuple[str, ...]
    excluded_parent_ids: tuple[str, ...]
    unseen_parent_ids: tuple[str, ...]


@dataclass(frozen=True)
class WindowNomination:
    chunk_index: int
    query_id: str
    issue_id: str
    query_text: str
    window: SourceWindow
    direct: bool
    targeted: bool
    group_ids: tuple[str, ...]
    lead_in: bool = False


@dataclass(frozen=True)
class ScoredWindow:
    nomination: WindowNomination
    raw_score: float
    semantic_eligible: bool


@dataclass(frozen=True)
class RerankResult:
    scored: tuple[ScoredWindow, ...]
    unseen: tuple[WindowNomination, ...]
    windows: tuple[SourceWindow, ...]
    groups: tuple[WindowGroup, ...]
    spent_pairs: int
    initial_pairs_after: int
    supplementary_pairs_after: int
    budget_exhausted: bool
    unseen_parent_ids: tuple[str, ...]
    warnings: tuple[str, ...]


def pool_candidates(batches: list[CandidateBatch] | tuple[CandidateBatch, ...],
                    max_parents: int = 48) -> CandidatePool:
    """Prioritize direct targets, then round-robin issues and their queries.

    Scores are retained per nomination and are never summed across queries.
    Every batch's direct parts are present before this explicit budget step.
    """
    if type(max_parents) is not int or not 0 < max_parents <= 48:
        raise ValueError("active parent budget must be in 1..48")
    nominations: OrderedDict[str, list[Candidate]] = OrderedDict()
    for batch in batches:
        for candidate in batch.candidates:
            nominations.setdefault(candidate.chunk_id, []).append(candidate)
    selected: list[str] = []
    chosen: set[str] = set()
    for priority in (2, 1, 0):
        issues: OrderedDict[str, OrderedDict[str, deque]] = OrderedDict()
        for batch in batches:
            queue = issues.setdefault(batch.issue_id, OrderedDict()).setdefault(batch.query_id, deque())
            queue.extend(c for c in batch.candidates if (2 if c.targeted else 1 if c.direct else 0) == priority)
        turns = {issue: deque(queries.values()) for issue, queries in issues.items()}
        while len(selected) < max_parents and any(turns.values()):
            for issue in issues:
                queues = turns[issue]
                while queues:
                    queue = queues.popleft()
                    while queue and queue[0].chunk_id in chosen:
                        queue.popleft()
                    if not queue:
                        continue
                    candidate = queue.popleft()
                    if queue:
                        queues.append(queue)
                    chosen.add(candidate.chunk_id)
                    selected.append(candidate.chunk_id)
                    break
                if len(selected) >= max_parents:
                    break
    excluded = tuple(chunk_id for chunk_id in nominations if chunk_id not in chosen)
    unseen = tuple(dict.fromkeys((*excluded, *(chunk_id for batch in batches for chunk_id in batch.unseen_parent_ids))))
    return CandidatePool(tuple(PooledParent(chunk_id, nominations[chunk_id][0].chunk_index,
                                             tuple(nominations[chunk_id])) for chunk_id in selected),
                         tuple(nominations), excluded, unseen)


def _window_order(windows: tuple[SourceWindow, ...], query: str, targets: list[tuple[int, int]],
                  leadins: list[tuple[int, int]]):
    words = set(re.findall(r"\w+", strip_accents(query).lower()))
    # Coverage tie order deliberately alternates early/late/middle windows.
    pending = list(range(len(windows)))
    coverage = []
    while pending:
        coverage.append(pending.pop(0))
        if pending:
            coverage.append(pending.pop())
    order = {index: rank for rank, index in enumerate(coverage)}
    def key(index):
        window = windows[index]
        targeted = any(window.start < end and start < window.end for start, end in targets)
        leadin = any(window.start < end and start < window.end for start, end in leadins)
        lexical = len(words & set(re.findall(r"\w+", strip_accents(window.text).lower())))
        return (not targeted, not leadin, -lexical, order[index])
    return sorted(range(len(windows)), key=key)


def _fair_window_order(nominations: list[WindowNomination]) -> list[WindowNomination]:
    """Preserve target/condition priority after deduplicating shared parents."""
    def priority(nomination):
        return (3 if nomination.targeted else 2 if nomination.lead_in
                else 1 if nomination.direct else 0)
    return [nomination for level in (3, 2, 1, 0)
            for nomination in _rotate_windows([item for item in nominations if priority(item) == level])]


def _rotate_windows(nominations: list[WindowNomination]) -> list[WindowNomination]:
    issues = OrderedDict()
    for nomination in nominations:
        issues.setdefault(nomination.issue_id, OrderedDict()).setdefault(
            nomination.query_id, OrderedDict()).setdefault(
                nomination.window.parent_chunk_id, deque()).append(nomination)
    turns = {issue: deque(deque(parents.values()) for parents in queries.values())
             for issue, queries in issues.items()}
    result = []
    while any(turns.values()):
        for issue in issues:
            queues = turns[issue]
            if queues:
                parents = queues.popleft()
                queue = parents.popleft()
                result.append(queue.popleft())
                if queue:
                    parents.append(queue)
                if parents:
                    queues.append(parents)
    return result


_EXPLICIT_TYPES = {
    "ctcp": r"cong\s+ty\s+co\s+phan|ctcp",
    "tnhh_mtv": r"(?:tnhh|trach\s+nhiem\s+huu\s+han)\s+(?:mot|1)\s+thanh\s+vien|tnhh\s*(?:1|m)\s*tv",
    "tnhh_2tv": r"(?:tnhh|trach\s+nhiem\s+huu\s+han)\s+(?:tu\s+)?(?:hai|2|nhieu|[3-9]|[1-9]\d+)\s+thanh\s+vien|tnhh\s*2\s*tv",
    "dnnn": r"doanh\s+nghiep\s+nha\s+nuoc|dnnn",
    "hop_danh": r"cong\s+ty\s+hop\s+danh",
    "dntn": r"doanh\s+nghiep\s+tu\s+nhan|dntn",
    "nhom_cty": r"tap\s+doan\s+kinh\s+te|nhom\s+cong\s+ty",
}


def _reference_targets(parent, references, groups):
    targets = []
    for reference in references:
        if not reference.khoan:
            if reference.diem:
                points = list(re.finditer(r"(?m)^\s*([a-zđ]|dd)\)\s+", parent["text"]))
                for index, point in enumerate(points):
                    if point.group(1).replace("dd", "đ") == reference.diem:
                        end = points[index + 1].start() if index + 1 < len(points) else len(parent["text"])
                        targets.append((point.start(), end))
            continue
        for group in groups:
            for unit in group.units:
                if unit.khoan != reference.khoan:
                    continue
                if reference.diem:
                    points = list(re.finditer(r"(?m)^\s*([a-zđ]|dd)\)\s+", unit.text))
                    for index, point in enumerate(points):
                        if point.group(1).replace("dd", "đ") == reference.diem:
                            end = points[index + 1].start() if index + 1 < len(points) else len(unit.text)
                            targets.append((unit.start + point.start(), unit.start + end))
                else:
                    targets.append((unit.start, unit.end))
    return list(dict.fromkeys(targets))

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

        self.valid = np.array([c["hieu_luc"] and c["vai_tro"] in {"chinh", "sua_doi"} for c in chunks])
        self.lh = np.array([c.get("loai_hinh") or "" for c in chunks])
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
        m = RE_Q_DIEU.search(q.lower())
        if not m:
            return [], None
        target = m.group(1).replace("dd", "đ")
        doc = next((d for pat, d in self.doc_aliases if re.search(pat, qa)), None)
        hits = [i for (d, dieu), idx in self.by_article.items()
                if dieu == target and (doc is None or d == doc) for i in idx]
        if not hits:
            target_na = strip_accents(target)
            hits = [i for (d, dieu), idx in self.by_article.items()
                    if strip_accents(dieu) == target_na and (doc is None or d == doc) for i in idx]
        ambiguous = doc is None and len({self.chunks[i]["doc_id"] for i in hits}) > 1
        return hits, ("ambiguous" if ambiguous else None)

    def topk(self, scores: np.ndarray, n: int):
        scores = np.where(self.valid, scores, -np.inf)
        return [int(i) for i in np.argsort(-scores)[:n] if np.isfinite(scores[i])]

    def article_lookup_many(self, question: str) -> list[dict]:
        """All original parent parts for locally resolved pairs, in corpus order.

        This additive API does not filter repeal/history lookups or apply a
        budget. The resolver separately exposes unresolved reference records.
        Legacy article_lookup/retrieve remain unchanged.
        """
        pairs: set[tuple[str, str]] = set()
        for reference in resolve_article_references(question):
            if reference.doc_id is None or reference.ambiguous:
                continue
            pair = (reference.doc_id, reference.dieu)
            if pair in self.by_article:
                pairs.add(pair)
            else:
                # Preserve legacy unaccented suffix fallback only when no exact
                # article exists; never spread a reference across documents.
                pairs.update((doc, dieu) for doc, dieu in self.by_article
                             if doc == reference.doc_id and strip_accents(dieu) == strip_accents(reference.dieu))
        return [chunk for chunk in self.chunks if (chunk["doc_id"], chunk["dieu"]) in pairs]

    @staticmethod
    def rrf(rank_lists: list[list[int]], k: int = 60):
        s = {}
        for r in rank_lists:
            for pos, i in enumerate(r):
                s[i] = s.get(i, 0.0) + 1.0 / (k + pos + 1)
        return sorted(s, key=s.get, reverse=True)

    def candidate_batch(self, query: str, *, query_id: str, issue_id: str,
                        original_question: str, references=None, company_type: str | None = None,
                        company_type_quote: str | None = None, per_channel: int = 20,
                        per_query: int = 12) -> CandidateBatch:
        """Nominate parents without final slicing/reranking; no request state.

        Direct parts are additional to the semantic per-query budget. Callers
        supply original-question references (not rewritten-query references).
        A type branch needs a literal original quote naming the supplied type.
        """
        if (type(per_channel) is not int or type(per_query) is not int or
                not 0 < per_channel <= 20 or not 0 < per_query <= 12):
            raise ValueError("candidate limits must be in 1..20 and 1..12")
        if not query_id or not issue_id:
            raise ValueError("query and issue IDs are required")
        if company_type is not None:
            pattern = _EXPLICIT_TYPES.get(company_type)
            if (not pattern or not company_type_quote or company_type_quote not in original_question or
                    not re.search(r"\b(?:" + pattern + r")\b", strip_accents(company_type_quote).lower()) or
                    detect_loai_hinh(company_type_quote) != company_type):
                raise ValueError("type scope requires an explicit original-question quote")
        refs = tuple(resolve_article_references(original_question) if references is None else references)
        for reference in refs:
            if original_question[reference.start:reference.end] != reference.quote:
                raise ValueError("reference quote must belong to original question")
        qn = self.norm_query(query)
        dense = np.asarray(self.emb @ self.embedder.encode(qn, normalize_embeddings=True))
        channel_scores = OrderedDict((
            ("dense", dense), ("bm25", np.asarray(self.bm25.get_scores(tok(qn)))),
            ("bm25_noacc", np.asarray(self.bm25_noacc.get_scores(tok_noacc(qn)))),
        ))
        if company_type is not None:
            compatible = np.array([not is_conflict(label or None, company_type) for label in self.lh])
            channel_scores["dense_type"] = np.where(compatible, dense, -np.inf)
        ranks, available, channel_unseen = OrderedDict(), {}, {}
        for channel, scores in channel_scores.items():
            eligible = np.where(self.valid, scores, -np.inf)
            available[channel] = int(np.isfinite(eligible).sum())
            ranks[channel] = [int(index) for index in np.argsort(-eligible, kind="stable")[:per_channel]
                              if np.isfinite(eligible[index])]
            channel_unseen[channel] = max(0, available[channel] - len(ranks[channel]))
        fused = self.rrf(list(ranks.values()), k=60)
        scores = {index: sum(1.0 / (60 + rank.index(index) + 1) for rank in ranks.values() if index in rank)
                  for index in fused}
        direct_refs: dict[int, list[ArticleReference]] = {}
        warnings = []
        for reference in refs:
            if reference.ambiguous or reference.doc_id is None:
                warnings.append("reference_unresolved")
                continue
            indices = self.by_article.get((reference.doc_id, reference.dieu))
            if indices is None:
                indices = [index for (doc, article), parts in self.by_article.items()
                           if doc == reference.doc_id and strip_accents(article) == strip_accents(reference.dieu)
                           for index in parts]
            if not indices:
                warnings.append("reference_parent_unavailable")
            for index in indices:
                direct_refs.setdefault(index, []).append(reference)
        semantic = fused[:per_query]
        selected = list(dict.fromkeys([*direct_refs, *semantic]))
        candidates = []
        for index in selected:
            parent = self.chunks[index]
            matching = tuple(direct_refs.get(index, ()))
            targeted = bool(_reference_targets(parent, matching, source_groups(parent, ())))
            candidates.append(Candidate(parent["chunk_id"], index, query_id, issue_id, query,
                                        fused.index(index) + 1 if index in scores else None, scores.get(index),
                                        tuple(channel for channel, rank in ranks.items() if index in rank),
                                        bool(matching), targeted, matching))
        unseen = tuple(self.chunks[index]["chunk_id"] for index in fused[per_query:] if index not in direct_refs)
        return CandidateBatch(query_id, issue_id, tuple(candidates), unseen, available, channel_unseen,
                              refs, tuple(dict.fromkeys(warnings)))

    def rerank_pool(self, pool: CandidatePool, *, tokenizer=None, supplementary: bool = False,
                    initial_pairs_used: int = 0, supplementary_pairs_used: int = 0,
                    initial_capacity: int = 72, supplementary_capacity: int = 24,
                    remaining_pairs: int | None = None, query_tokens: int = 96,
                    passage_tokens: int = 384, overlap_tokens: int = 64,
                    pair_tokens: int = 512, exclude_scored=()) -> RerankResult:
        """Score RAM windows using raw predict values and caller-owned counters.

        Unsupported tokenizers/pairs leave explicit unseen parents/windows.
        No tokenizer/model is loaded and no counter is stored on the retriever.
        """
        integer_values = (initial_pairs_used, supplementary_pairs_used, initial_capacity,
                          supplementary_capacity, query_tokens, passage_tokens, overlap_tokens, pair_tokens)
        if any(type(value) is not int for value in integer_values):
            raise ValueError("rerank capacities must be integers")
        if (not 0 <= initial_pairs_used <= initial_capacity <= 72 or
                not 0 <= supplementary_pairs_used <= supplementary_capacity <= 24 or
                not 0 < query_tokens <= 96 or not 0 < passage_tokens <= 384 or
                not 0 <= overlap_tokens < passage_tokens or not 0 < pair_tokens <= 512):
            raise ValueError("invalid rerank budget")
        capacity = (supplementary_capacity - supplementary_pairs_used if supplementary
                    else initial_capacity - initial_pairs_used)
        if remaining_pairs is not None:
            if type(remaining_pairs) is not int or not 0 <= remaining_pairs <= 96:
                raise ValueError("remaining pairs must be in 0..96")
            capacity = min(capacity, remaining_pairs)
        excluded_keys = set()
        for key in exclude_scored:
            if (not isinstance(key, (tuple, list)) or len(key) != 3 or
                    any(not isinstance(value, str) or not value for value in key)):
                raise ValueError("exclude_scored keys must be issue/query/window ID triples")
            excluded_keys.add(tuple(key))
        tokenizer = tokenizer if tokenizer is not None else getattr(self.reranker, "tokenizer", None)
        windows, groups, nominations, warnings, unsupported_parents = [], [], [], [], []
        scoring_queries = {}
        for parent in pool.parents:
            chunk = self.chunks[parent.chunk_index]
            if chunk["chunk_id"] != parent.chunk_id:
                raise ValueError("pool parent identity mismatch")
            try:
                views = build_source_windows(chunk, tokenizer, passage_tokens=passage_tokens,
                                             overlap_tokens=overlap_tokens)
                if not views:
                    warnings.append("source_has_no_tokens")
                    unsupported_parents.append(parent.chunk_id)
                linked = source_groups(chunk, views)
                windows.extend(views)
                groups.extend(linked)
                for candidate in parent.nominations:
                    key = (candidate.query_id, candidate.issue_id)
                    scoring_queries[key] = prepare_query(tokenizer, candidate.query_text, query_tokens)
                    targets = _reference_targets(chunk, candidate.references, linked)
                    leadins = list(dict.fromkeys((unit.start, unit.end) for group in linked
                        if any(member.khoan for member in group.units)
                        for unit in group.units if unit.khoan is None))
                    for index in _window_order(views, candidate.query_text, targets, leadins):
                        view = views[index]
                        nominations.append(WindowNomination(parent.chunk_index, candidate.query_id,
                            candidate.issue_id, candidate.query_text, view, candidate.direct,
                            any(view.start < end and start < view.end for start, end in targets),
                            tuple(group.group_id for group in linked if view.window_id in group.window_ids),
                            any(view.start < end and start < view.end for start, end in leadins)))
            except UnsupportedTokenizerError as error:
                warnings.append(error.reason)
                unsupported_parents.append(parent.chunk_id)
        # Fair issue/parent allocation applies even when a long source has many
        # views or several near-paraphrase query nominations.
        ordered = _fair_window_order([item for item in nominations
            if (item.issue_id, item.query_id, item.window.window_id) not in excluded_keys])
        pairs, selected, unseen = [], [], []
        for nomination in ordered:
            query = scoring_queries[(nomination.query_id, nomination.issue_id)]
            try:
                pair_size = pair_token_count(tokenizer, query, nomination.window.text)
            except UnsupportedTokenizerError as error:
                warnings.append(error.reason)
                unseen.append(nomination)
                continue
            if pair_size > pair_tokens:
                warnings.append("pair_token_limit_exceeded")
                unseen.append(nomination)
            elif len(selected) >= capacity:
                unseen.append(nomination)
            else:
                selected.append(nomination)
                pairs.append((query, nomination.window.text))
        raw = np.asarray(self.reranker.predict(pairs)).reshape(-1) if pairs else np.array([])
        if len(raw) != len(selected) or not np.isfinite(raw).all():
            raise ValueError("reranker returned invalid raw scores")
        scored = tuple(ScoredWindow(nomination, float(score), float(score) >= self.rerank_min)
                       for nomination, score in zip(selected, raw))
        spent = len(selected)
        return RerankResult(scored, tuple(unseen), tuple(windows), tuple(groups), spent,
                            initial_pairs_used + (0 if supplementary else spent),
                            supplementary_pairs_used + (spent if supplementary else 0),
                            bool(unseen) and spent >= capacity,
                            tuple(dict.fromkeys((*pool.unseen_parent_ids, *unsupported_parents,
                                                 *(item.window.parent_chunk_id for item in unseen)))),
                            tuple(dict.fromkeys(warnings)))

    def retrieve(self, q: str, n_cand: int = 20, n_rerank: int = 8, top: int = 5):
        q_lh = detect_loai_hinh(q)
        qn = self.norm_query(q)
        qv = self.embedder.encode(qn, normalize_embeddings=True)
        dense = self.emb @ qv
        lists = [
            self.topk(dense, n_cand),
            self.topk(self.bm25.get_scores(tok(qn)), n_cand),
            self.topk(self.bm25_noacc.get_scores(tok_noacc(qn)), n_cand),
        ]
        if q_lh:
            lists.append(self.topk(np.where(self.lh == q_lh, dense, -np.inf), n_cand))
        cand = self.rrf(lists)[:n_rerank]

        sc = self.reranker.predict([(qn, self.chunks[i]["text"]) for i in cand]) if cand else np.array([])
        order = np.argsort(-sc)
        if q_lh:
            # Nguồn đạt ngưỡng luôn được xét trước, rồi mới ưu tiên loại hình.
            order = sorted(order, key=lambda j: (
                sc[j] < self.rerank_min,
                is_conflict(self.chunks[cand[j]].get("loai_hinh"), q_lh),
            ))
        ranked = [(cand[j], float(sc[j])) for j in order[:top]]
        max_score = max((s for _, s in ranked), default=0.0)
        direct, flag = self.article_lookup(qn)
        hits = list(dict.fromkeys([i for i in direct[:4]] + [i for i, s in ranked if s >= self.rerank_min]))

        return {
            "hits": hits,
            "max_score": max_score,
            "low_confidence": not direct and (not ranked or max_score < self.rerank_min),
            "flag": flag,
            "loai_hinh": q_lh,
        }
