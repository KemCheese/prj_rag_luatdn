from typing import Literal, get_args


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
