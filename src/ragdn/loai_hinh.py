import re
from typing import Literal, get_args

from ragdn.index import strip_accents


LoaiHinh = Literal["tnhh_2tv", "tnhh_mtv", "dnnn", "ctcp", "hop_danh", "dntn", "nhom_cty"]

LABELS: dict[str, str] = {
    "tnhh_2tv": "Công ty TNHH hai thành viên trở lên",
    "tnhh_mtv": "Công ty TNHH một thành viên",
    "dnnn": "Doanh nghiệp nhà nước",
    "ctcp": "Công ty cổ phần",
    "hop_danh": "Công ty hợp danh",
    "dntn": "Doanh nghiệp tư nhân",
    "nhom_cty": "Nhóm công ty",
}
assert set(LABELS) == set(get_args(LoaiHinh))

TUONG_THICH: dict[str, frozenset[str]] = {
    "dnnn": frozenset({"tnhh_mtv", "tnhh_2tv", "ctcp"}),
    "nhom_cty": frozenset(get_args(LoaiHinh)),
}


def is_conflict(chunk_lh: str | None, q_lh: str | None) -> bool:
    """Kiểm tra loại hình của nguồn có xung đột với loại hình được hỏi hay không."""
    return bool(q_lh and chunk_lh and chunk_lh != q_lh and chunk_lh not in TUONG_THICH.get(q_lh, ()))


# Tên loại hình được ưu tiên trước các dấu hiệu về cơ quan hoặc chức danh.
_TEN = {
    "tnhh_2tv": [
        r"(?:tnhh|trach\s+nhiem\s+huu\s+han)\s+(?:tu\s+)?(?:hai|2)\s+thanh\s+vien",
        r"(?:tnhh|trach\s+nhiem\s+huu\s+han)\s+(?:nhieu|[3-9]|[1-9]\d+)\s+thanh\s+vien",
        r"tnhh\s*2\s*tv",
    ],
    "tnhh_mtv": [
        r"(?:tnhh|trach\s+nhiem\s+huu\s+han)\s+(?:mot|1)\s+thanh\s+vien",
        r"tnhh\s*(?:1|m)\s*tv",
    ],
    "dnnn": [r"doanh\s+nghiep\s+nha\s+nuoc", r"dnnn"],
    "ctcp": [r"cong\s+ty\s+co\s+phan", r"ctcp"],
    "hop_danh": [r"cong\s+ty\s+hop\s+danh"],
    "dntn": [r"doanh\s+nghiep\s+tu\s+nhan", r"dntn"],
    "nhom_cty": [r"tap\s+doan\s+kinh\s+te", r"nhom\s+cong\s+ty"],
}
_DAU_HIEU = {
    "ctcp": [
        r"dai\s+hoi\s+dong\s+co\s+dong",
        r"dhdcd",
        r"hoi\s+dong\s+quan\s+tri",
        r"hdqt",
        r"co\s+phieu",
    ],
    "hop_danh": [r"thanh\s+vien\s+hop\s+danh"],
    "dnnn": [r"co\s+quan\s+dai\s+dien\s+chu\s+so\s+huu"],
    "nhom_cty": [r"cong\s+ty\s+me", r"cong\s+ty\s+con"],
}
_TIERS = [
    {loai_hinh: re.compile(r"\b(?:" + "|".join(patterns) + r")\b") for loai_hinh, patterns in tier.items()}
    for tier in (_TEN, _DAU_HIEU)
]


def detect_loai_hinh(question: str) -> str | None:
    """Trả mã duy nhất ở tầng tên, hoặc tầng dấu hiệu khi không có tên; còn lại trả None."""
    normalized = strip_accents(question).lower()
    for tier in _TIERS:
        found = {loai_hinh for loai_hinh, pattern in tier.items() if pattern.search(normalized)}
        if found:
            return found.pop() if len(found) == 1 else None
    return None
