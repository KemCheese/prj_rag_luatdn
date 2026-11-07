"""RAM-only reranker windows and mechanically linked source groups.

Token counts here belong exclusively to the supplied reranker tokenizer. They
are never an estimate of the reader model's context capacity.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


class UnsupportedTokenizerError(RuntimeError):
    def __init__(self, reason: str = "tokenizer_offsets_unsupported"):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class SourceWindow:
    window_id: str
    parent_chunk_id: str
    start: int
    end: int
    text: str
    token_count: int


@dataclass(frozen=True)
class SourceUnit:
    unit_id: str
    start: int
    end: int
    text: str
    khoan: str | None = None


@dataclass(frozen=True)
class WindowGroup:
    group_id: str
    parent_chunk_id: str
    units: tuple[SourceUnit, ...]
    window_ids: tuple[str, ...]
    atomic: bool = True


def encoded(tokenizer, text: str, *, offsets: bool = False, pair: str | None = None):
    if tokenizer is None or not callable(tokenizer):
        raise UnsupportedTokenizerError("reranker_tokenizer_missing")
    kwargs = {"add_special_tokens": pair is not None, "truncation": False}
    if offsets:
        kwargs["return_offsets_mapping"] = True
        # Full-text offset scans intentionally precede window/query limits;
        # they do not send this untruncated sequence to the model.
        kwargs["verbose"] = False
    try:
        result = tokenizer(text, pair, **kwargs) if pair is not None else tokenizer(text, **kwargs)
    except (TypeError, NotImplementedError) as error:
        raise UnsupportedTokenizerError() from error
    if "input_ids" not in result:
        raise UnsupportedTokenizerError("tokenizer_input_ids_missing")
    ids = result["input_ids"]
    if not isinstance(ids, (list, tuple)) or any(not isinstance(item, int) for item in ids):
        raise UnsupportedTokenizerError("tokenizer_single_sequence_required")
    if offsets:
        spans = result.get("offset_mapping")
        if not isinstance(spans, (list, tuple)) or len(spans) != len(ids):
            raise UnsupportedTokenizerError()
        previous = 0
        for span in spans:
            if (not isinstance(span, (list, tuple)) or len(span) != 2 or
                    any(not isinstance(value, int) for value in span) or
                    not 0 <= span[0] < span[1] <= len(text) or span[0] < previous):
                raise UnsupportedTokenizerError("tokenizer_offsets_invalid")
            previous = span[0]
    return result


def prepare_query(tokenizer, query: str, max_tokens: int = 96) -> str:
    if type(max_tokens) is not int or not 0 < max_tokens <= 96:
        raise ValueError("query token budget must be in 1..96")
    result = encoded(tokenizer, query, offsets=True)
    if len(result["input_ids"]) <= max_tokens:
        return query
    # Only the scoring query is shortened, at a tokenizer-provided boundary.
    # Original question/query remain in request provenance, unchanged.
    for count in range(max_tokens, 0, -1):
        view = query[:result["offset_mapping"][count - 1][1]]
        if len(encoded(tokenizer, view)["input_ids"]) <= max_tokens:
            return view
    raise UnsupportedTokenizerError("query_token_boundary_unstable")


def build_source_windows(parent: dict, tokenizer, *, passage_tokens: int = 384,
                         overlap_tokens: int = 64) -> tuple[SourceWindow, ...]:
    if (type(passage_tokens) is not int or type(overlap_tokens) is not int or
            not 0 < passage_tokens <= 384 or not 0 <= overlap_tokens < passage_tokens):
        raise ValueError("invalid passage/overlap budget")
    text = parent["text"]
    result = encoded(tokenizer, text, offsets=True)
    spans = result["offset_mapping"]
    if not spans:
        return ()
    windows = []
    index = 0
    while index < len(spans):
        stop = min(len(spans), index + passage_tokens)
        # Prefer a reliable line/sentence end near the cap, without sacrificing
        # overlap progress. Clause groups below retain the complete conditions.
        floor = max(index + overlap_tokens + 1, stop - min(64, passage_tokens // 4))
        for candidate in range(stop, floor - 1, -1):
            char_end = spans[candidate - 1][1]
            if candidate == len(spans) or re.search(r"[.!?;]\s*$|\n\s*$", text[max(0, char_end - 3):char_end + 1]):
                stop = candidate
                break
        start = 0 if index == 0 else spans[index][0]
        end = len(text) if stop == len(spans) else spans[stop - 1][1]
        view = text[start:end]
        count = len(encoded(tokenizer, view)["input_ids"])
        if count > passage_tokens:
            # Tokenization of a substring can differ from parent tokenization;
            # reject instead of relying on silent CrossEncoder truncation.
            raise UnsupportedTokenizerError("passage_token_boundary_unstable")
        windows.append(SourceWindow(f"{parent['chunk_id']}@{start}:{end}", parent["chunk_id"],
                                    start, end, view, count))
        if stop == len(spans):
            break
        index = stop - overlap_tokens
    return tuple(windows)


def pair_token_count(tokenizer, query: str, passage: str) -> int:
    """Actual pair encoding, including this tokenizer's special tokens."""
    return len(encoded(tokenizer, query, pair=passage)["input_ids"])


def source_groups(parent: dict, windows: tuple[SourceWindow, ...]) -> tuple[WindowGroup, ...]:
    """Group full clauses, introductory text and explicit intra-parent dependencies.

    These are structural/lexical links, not a legal completeness judgment. An
    unstructured parent remains one atomic group instead of losing exceptions.
    """
    text, chunk_id = parent["text"], parent["chunk_id"]
    starts = list(re.finditer(r"(?m)^\s*(\d+(?:dd|[a-zđ])?)\.\s+", text))
    units: list[SourceUnit] = []
    if starts and starts[0].start() > 0:
        end = starts[0].start()
        units.append(SourceUnit(f"{chunk_id}:u0", 0, end, text[:end]))
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        units.append(SourceUnit(f"{chunk_id}:u{len(units)}", match.start(), end,
                                text[match.start():end], match.group(1).replace("dd", "đ")))
    if not units and text:
        units = [SourceUnit(f"{chunk_id}:u0", 0, len(text), text)]
    clauses = [unit for unit in units if unit.khoan]
    # Amendment/quoted provisions can repeat a clause number. A local clause
    # reference cannot safely select one occurrence, so preserve every unit
    # and source window as one atomic parent group instead of guessing.
    clause_numbers = [unit.khoan for unit in clauses]
    if len(set(clause_numbers)) != len(clause_numbers):
        return (WindowGroup(f"{chunk_id}:g0", chunk_id, tuple(units),
                            tuple(window.window_id for window in windows)),)
    intro = tuple(unit for unit in units if not unit.khoan)
    if not clauses:
        clauses = units
        intro = ()
    links = {unit.unit_id: {unit.unit_id} for unit in clauses}
    by_clause = {unit.khoan: unit for unit in clauses if unit.khoan}
    for index, unit in enumerate(clauses):
        targets = {by_clause[match.group(1).replace("dd", "đ")].unit_id for match in re.finditer(
            r"\b(?:khoản|khoan)\s+(\d+(?:[a-zđ]|dd)?)\b", unit.text.lower())
            if match.group(1).replace("dd", "đ") in by_clause}
        body = re.sub(r"^\s*\d+(?:dd|[a-zđ])?\.\s+", "", unit.text.lower())
        if index and re.match(r"(?:trừ|tru|ngoại trừ|ngoai tru|tuy nhiên|tuy nhien)\b", body):
            targets.add(clauses[index - 1].unit_id)
        for target in targets:
            links[unit.unit_id].add(target)
            links[target].add(unit.unit_id)
    groups, visited = [], set()
    for unit in clauses:
        if unit.unit_id in visited:
            continue
        component, pending = set(), [unit.unit_id]
        while pending:
            current = pending.pop()
            if current not in component:
                component.add(current)
                pending.extend(links[current] - component)
        visited.update(component)
        members = tuple(sorted((*intro, *(u for u in clauses if u.unit_id in component)), key=lambda u: u.start))
        ids = tuple(w.window_id for w in windows if any(w.start < u.end and u.start < w.end for u in members))
        groups.append(WindowGroup(f"{chunk_id}:g{len(groups)}", chunk_id, members, ids))
    return tuple(groups)
