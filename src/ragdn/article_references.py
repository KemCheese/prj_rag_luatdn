"""Conservative, source-span-preserving document/article references.

This module never expands references from legal knowledge or selects a default
document. Unsupported/competing local bindings remain visible to the caller.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
import unicodedata

from pydantic import BaseModel, ConfigDict, Field
import yaml

from ragdn.config import ALIASES_PATH, MANIFEST_PATH
from ragdn.manifest import load_manifest


class ArticleReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str | None = None
    dieu: str
    khoan: str | None = None
    diem: str | None = None
    quote: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    ambiguous: bool = False
    reason: str | None = None
    role: str | None = None
    amendment_target: str | None = None


@dataclass(frozen=True)
class _Mention:
    start: int
    end: int
    doc_id: str | None


@dataclass(frozen=True)
class _Group:
    start: int
    end: int
    numbers: tuple[tuple[int, int], ...]
    khoan: tuple[int, int] | None = None
    diem: tuple[int, int] | None = None


_NUMBER = r"\d+(?:dd|[a-z])?(?![\w%/])"
_ARTICLE = re.compile(r"\bdieu\s+(" + _NUMBER + r")")
_NEXT = re.compile(r"\s*(?:,\s*(?:va\s+)?|va\s+)(?:dieu\s+)?(" + _NUMBER + r")")
_QUALIFIER = re.compile(
    r"(?:\bdiem\s+(?P<diem>dd|[a-z])\s*,?\s*)?"
    r"(?:\bkhoan\s+(?P<khoan>\d+(?:dd|[a-z])?)\s*,?\s*)?$"
)
_AFTER = re.compile(r"\s*(?:(?:cua|tai|trong|theo)\s+)?$")
_BEFORE = re.compile(r"\s*[,\-:]?\s*(?:(?:tai|theo|cac|quy dinh tai)\s+)*$")
_DOC_JOIN = re.compile(r"\s*(?:,|va|hoac)\s*$")


def _fold(text: str) -> tuple[str, list[int]]:
    """Accent folding with original character indices, including decomposed input."""
    folded: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        for base in unicodedata.normalize("NFD", char.lower().replace("đ", "d")):
            if unicodedata.category(base) != "Mn":
                folded.append(base)
                positions.append(index)
    positions.append(len(text))
    return "".join(folded), positions


def _documents(manifest) -> list[dict]:
    if manifest is None:
        manifest = load_manifest(MANIFEST_PATH)
    if hasattr(manifest, "model_dump"):
        manifest = manifest.model_dump()
    if isinstance(manifest, Mapping):
        if "documents" in manifest:
            manifest = manifest["documents"]
        elif "doc_id" in manifest:
            manifest = [manifest]
        else:
            raise TypeError("manifest must contain documents or a document's doc_id")
    if not isinstance(manifest, Sequence) or isinstance(manifest, (str, bytes)):
        raise TypeError("manifest must be a Manifest, document mapping, or document sequence")
    result = []
    for document in manifest:
        if hasattr(document, "model_dump"):
            document = document.model_dump()
        if not isinstance(document, Mapping) or not isinstance(document.get("doc_id"), str):
            raise TypeError("each document must have a string doc_id")
        result.append(dict(document))
    return result


def _document_mentions(text: str, documents: list[dict]) -> list[_Mention]:
    supported = {doc["doc_id"] for doc in documents}
    candidates: list[_Mention] = []
    if ALIASES_PATH.exists():
        aliases = yaml.safe_load(ALIASES_PATH.read_text(encoding="utf-8")) or {}
        for alias in aliases.get("aliases", []):
            if alias["doc_id"] in supported:
                candidates.extend(_Mention(m.start(), m.end(), alias["doc_id"])
                                  for m in re.finditer(alias["pattern"], text))
    for document in documents:
        for key in ("ten_vb", "ten_ngan", "so_hieu"):
            value = document.get(key)
            if isinstance(value, str) and value:
                folded, _ = _fold(value)
                pattern = r"(?<!\w)" + re.escape(folded) + r"(?!\w)"
                candidates.extend(_Mention(m.start(), m.end(), document["doc_id"])
                                  for m in re.finditer(pattern, text))
    # A complete amendment title contains its target's name; the containing
    # supported title wins rather than introducing a second document mention.
    accepted: list[_Mention] = []
    for candidate in sorted(candidates, key=lambda m: (-(m.end - m.start), m.start, m.doc_id)):
        overlaps = [m for m in accepted if m.start < candidate.end and candidate.start < m.end]
        if not overlaps:
            accepted.append(candidate)
        elif all(m.doc_id == candidate.doc_id for m in overlaps):
            accepted = [m for m in accepted if m not in overlaps]
            accepted.append(_Mention(min([candidate.start] + [m.start for m in overlaps]),
                                     max([candidate.end] + [m.end for m in overlaps]), candidate.doc_id))
        elif not any(m.start <= candidate.start and candidate.end <= m.end
                     and (m.start, m.end) != (candidate.start, candidate.end) for m in overlaps):
            # Identical supported aliases assigned to different documents are
            # a genuine conflict, not an invitation to choose the first one.
            accepted.append(candidate)
    ordered = sorted(accepted, key=lambda m: (m.start, m.end))
    merged: list[_Mention] = []
    for mention in ordered:
        if merged and merged[-1].doc_id == mention.doc_id and re.fullmatch(
                r"\s*(?:so\s*)?", text[merged[-1].end:mention.start]):
            previous = merged.pop()
            merged.append(_Mention(previous.start, mention.end, mention.doc_id))
        else:
            merged.append(mention)
    qualified: list[_Mention] = []
    by_id = {document["doc_id"]: document for document in documents}
    for mention in merged:
        document = by_id[mention.doc_id]
        identifier, _ = _fold(document.get("so_hieu") or "")
        tail = text[mention.end:]
        end = mention.end
        # A generic alias cannot silently swallow a different edition/number.
        # Extend a confirmed qualifier into the provenance span, or reject the
        # binding when the explicitly supplied qualifier is unsupported.
        full_number = re.match(r"\s+(?:so\s*)?(\d+/(?:19|20)\d{2}/[\w-]+)\b", tail)
        slash_number = re.match(r"(/(?:19|20)\d{2}/[\w-]+)\b", tail)
        year = re.match(r"\s*(?:\(\s*)?(?:nam\s+)?((?:19|20)\d{2})\b\s*\)?", tail)
        short_number = re.match(r"\s+so\s+(\d+)\b", tail)
        if full_number:
            if full_number.group(1) != identifier:
                qualified.append(_Mention(mention.start, end + full_number.end(), None))
                continue
            end += full_number.end()
        elif slash_number:
            prefix = re.search(r"\d+$", text[mention.start:mention.end])
            if not prefix or prefix.group() + slash_number.group(1) != identifier:
                qualified.append(_Mention(mention.start, end + slash_number.end(), None))
                continue
            end += slash_number.end()
        elif year:
            document_year = re.search(r"/((?:19|20)\d{2})/", identifier)
            if not document_year or year.group(1) != document_year.group(1):
                qualified.append(_Mention(mention.start, end + year.end(), None))
                continue
            end += year.end()
        elif short_number:
            if short_number.group(1) != identifier.split("/", 1)[0]:
                qualified.append(_Mention(mention.start, end + short_number.end(), None))
                continue
            end += short_number.end()
        qualified.append(_Mention(mention.start, end, mention.doc_id))
    # Keep unsupported document mentions in local clusters. Dropping an unknown
    # member would incorrectly turn "known law and unknown law" into certainty.
    for marker in re.finditer(r"\b(?:bo luat|luat|nghi dinh|nd|thong tu|quyet dinh)\b", text):
        if any(m.start <= marker.start() < m.end for m in qualified):
            continue
        stop = re.search(r"[,;:?.()\n]|\b(?:va|hoac)\b|\bdieu\s+\d", text[marker.end():])
        end = marker.end() + stop.start() if stop else len(text)
        following = [m.start for m in qualified if m.start >= marker.end()]
        if following:
            end = min(end, min(following))
        while end > marker.end() and text[end - 1].isspace():
            end -= 1
        qualified.append(_Mention(marker.start(), end, None))
    # Bare document numbers are also supported names. Unknown numbers in a
    # conjunction must participate in the same conflict check as named laws.
    for identifier_match in re.finditer(r"\b\d+/(?:19|20)\d{2}/[a-z0-9-]+\b", text):
        if not any(m.start <= identifier_match.start() < m.end for m in qualified):
            qualified.append(_Mention(identifier_match.start(), identifier_match.end(), None))
    return sorted(qualified, key=lambda mention: (mention.start, mention.end))


def _groups(text: str) -> list[_Group]:
    groups: list[_Group] = []
    consumed = 0
    for match in _ARTICLE.finditer(text):
        if match.start() < consumed:
            continue
        numbers = [match.span(1)]
        end = match.end()
        while following := _NEXT.match(text, end):
            numbers.append(following.span(1))
            end = following.end()
        consumed = end
        qualifier = _QUALIFIER.search(text[:match.start()])
        start = match.start()
        clause = point = None
        if qualifier and qualifier.start() < match.start():
            start = qualifier.start()
            clause = qualifier.span("khoan") if qualifier.group("khoan") else None
            point = qualifier.span("diem") if qualifier.group("diem") else None
        groups.append(_Group(start, end, tuple(numbers), clause, point))
    return groups


def _depths(text: str) -> list[int]:
    result, depth = [], 0
    for char in text:
        result.append(depth)
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
    result.append(depth)
    return result


def _without_amendment_notes(gap: str) -> str:
    def replace(match: re.Match) -> str:
        note = match.group(0)
        if re.search(r"\b(?:sua doi|bo sung|hop nhat)\b", note) and not _ARTICLE.search(note):
            return " "
        return note
    return re.sub(r"\([^()]*\)", replace, gap)


def _cluster(mention: _Mention, mentions: list[_Mention], text: str, depths: list[int]) -> list[_Mention]:
    index = mentions.index(mention)
    left = right = index
    while left and depths[mentions[left - 1].start] == depths[mention.start] and _DOC_JOIN.fullmatch(
            text[mentions[left - 1].end:mentions[left].start]):
        left -= 1
    while right + 1 < len(mentions) and depths[mentions[right + 1].start] == depths[mention.start] and _DOC_JOIN.fullmatch(
            text[mentions[right].end:mentions[right + 1].start]):
        right += 1
    return mentions[left:right + 1]


def resolve_article_references(question: str, manifest=None) -> list[ArticleReference]:
    """Resolve explicit local references; unknown or conflicting documents stay unresolved.

    ``manifest`` accepts the existing Manifest, its dict, a single document
    metadata mapping or a sequence of those mappings/DocumentMeta objects.
    When omitted, the existing manifest and aliases are read at call time.
    Group references share an exact original quote; output order follows the
    question. ``role``/``amendment_target`` describe manifest metadata only.
    """
    documents = _documents(manifest)
    metadata = {document["doc_id"]: document for document in documents}
    folded, positions = _fold(question)
    mentions = _document_mentions(folded, documents)
    depths = _depths(folded)
    result: list[ArticleReference] = []

    def original(span: tuple[int, int] | None) -> str | None:
        return unicodedata.normalize("NFC", question[positions[span[0]]:positions[span[1]]]).lower().replace("dd", "đ") if span else None

    for group in _groups(folded):
        candidates = []
        for mention in mentions:
            if depths[mention.start] != depths[group.start]:
                continue
            if mention.end <= group.start and _BEFORE.fullmatch(_without_amendment_notes(folded[mention.end:group.start])):
                candidates.extend(_cluster(mention, mentions, folded, depths))
            elif mention.start >= group.end and _AFTER.fullmatch(_without_amendment_notes(folded[group.end:mention.start])):
                candidates.extend(_cluster(mention, mentions, folded, depths))
        doc_ids = {mention.doc_id for mention in candidates}
        doc_id = next(iter(doc_ids)) if len(doc_ids) == 1 else None
        reason = None if doc_id else "conflicting_documents" if len(doc_ids) > 1 else "document_unresolved"
        start = min([group.start] + [mention.start for mention in candidates])
        end = max([group.end] + [mention.end for mention in candidates])
        original_start, original_end = positions[start], positions[end]
        document = metadata.get(doc_id, {})
        for index, number in enumerate(group.numbers):
            result.append(ArticleReference(
                doc_id=doc_id, dieu=original(number),
                khoan=original(group.khoan) if index == 0 else None,
                diem=original(group.diem) if index == 0 else None,
                quote=question[original_start:original_end], start=original_start, end=original_end,
                ambiguous=doc_id is None, reason=reason, role=document.get("vai_tro"),
                amendment_target=document.get("sua_doi_cho"),
            ))
    return result
