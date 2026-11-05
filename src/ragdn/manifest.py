# src/ragdn/manifest.py
from itertools import pairwise
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from ragdn.loai_hinh import LoaiHinh


class CauTruc(BaseModel):
    chuong: str
    muc: str | None = None
    dieu: tuple[int, int]
    loai_hinh: LoaiHinh | None = None


class DocumentMeta(BaseModel):
    doc_id: str
    ten_vb: str
    ten_ngan: str
    so_hieu: str
    loai_vb: str
    vai_tro: str = "chinh"
    sua_doi_cho: str | None = None
    hop_nhat: bool = False
    hop_nhat_den: list[str] = Field(default_factory=list)
    ngay_ban_hanh: str | None = None
    ngay_hieu_luc: str | None = None
    trang_thai: str = "con_hieu_luc"
    so_dieu: int | None = None
    cau_truc: list[CauTruc] = Field(default_factory=list)
    file: str
    sha256: str = ""
    nguon_url: str = ""
    ngay_tai: str = ""
    parse: dict = Field(default_factory=dict)

    @field_validator("cau_truc")
    @classmethod
    def _pham_vi_hop_le(cls, sections: list[CauTruc]) -> list[CauTruc]:
        for section in sections:
            first, last = section.dieu
            if first > last:
                raise ValueError(f"cau_truc: phạm vi ngược {section.dieu}")
        for left, right in pairwise(sorted(sections, key=lambda section: section.dieu[0])):
            if left.dieu[1] >= right.dieu[0]:
                raise ValueError(f"cau_truc chồng lấn: {left.dieu} và {right.dieu}")
        return sections


class Manifest(BaseModel):
    corpus_version: str
    documents: list[DocumentMeta]

def load_manifest(path: Path | str) -> Manifest:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Manifest(**data)
