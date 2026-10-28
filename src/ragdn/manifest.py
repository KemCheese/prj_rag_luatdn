# src/ragdn/manifest.py
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DocumentMeta(BaseModel):
    doc_id: str
    ten_vb: str
    ten_ngan: str
    so_hieu: str
    loai_vb: str
    vai_tro: str = "chinh"
    hop_nhat: bool = False
    hop_nhat_den: list[str] = Field(default_factory=list)
    ngay_ban_hanh: str | None = None
    ngay_hieu_luc: str | None = None
    trang_thai: str = "con_hieu_luc"
    so_dieu: int | None = None
    file: str
    sha256: str = ""
    nguon_url: str = ""
    ngay_tai: str = ""
    parse: dict = Field(default_factory=dict)

class Manifest(BaseModel):
    corpus_version: str
    documents: list[DocumentMeta]

def load_manifest(path: Path | str) -> Manifest:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Manifest(**data)
