import pytest
from pydantic import ValidationError

from ragdn.chunk import check_corpus, chunk
from ragdn.config import MANIFEST_PATH
from ragdn.manifest import load_manifest
from ragdn.parse import parse


@pytest.fixture
def document():
    return next(doc for doc in load_manifest(MANIFEST_PATH).documents if doc.doc_id == "luat_dn_2020")


def test_chunk_uses_manifest_section_labels(document):
    lines = [
        "Chương II",
        "Điều 17a. Điều bổ sung",
        "1. Nội dung.",
        "Chương III",
        "Mục 1",
        "Điều 59. Nghị quyết",
        "1. Nội dung nghị quyết.",
        "Mục 2",
        "Điều 74. Công ty một thành viên",
        "1. Nội dung.",
        "Chương IV",
        "Điều 90. Quyền của thành viên",
        "1. Nội dung.",
    ]
    articles = parse([(line, 1) for line in lines], document.doc_id)
    chunks = chunk(articles, document.model_dump())

    assert [(c["dieu"], c["chuong"], c["muc"], c["loai_hinh"]) for c in chunks] == [
        ("17a", "II", None, None),
        ("59", "III", "1", "tnhh_2tv"),
        ("74", "III", "2", "tnhh_mtv"),
        ("90", "IV", None, "dnnn"),
    ]
    assert chunks[1]["chunk_id"] == "luat_dn_2020:d59:p0"
    assert chunks[1]["text"].splitlines() == [
        "Luật Doanh nghiệp | Chương III | Mục 1 | Điều 59. Nghị quyết",
        "1. Nội dung nghị quyết.",
    ]
    assert "| Mục " not in chunks[3]["text"]


def test_check_corpus_detects_wrong_chapter(document):
    chunks = [{
        "chunk_id": "luat_dn_2020:d90:p0",
        "doc_id": document.doc_id,
        "dieu": "90",
        "chuong": "III",
        "muc": None,
        "text": "Điều 90. Quyền của thành viên\n1. Nội dung.",
    }]
    meta = document.model_dump()
    expected = "luat_dn_2020:d90:p0: parse ra Chương III/Mục None, manifest khai báo Chương IV/Mục None"
    assert expected in check_corpus(chunks, [meta])
    chunks[0]["chuong"] = "IV"
    assert not any("parse ra" in error for error in check_corpus(chunks, [meta]))


@pytest.mark.parametrize(
    ("sections", "error"),
    [
        ([{"chuong": "III", "muc": 1, "dieu": [46, 73]}], "string_type"),
        ([{"chuong": "III", "dieu": [46, 73], "loai_hinh": "TNHH_2TV"}], "literal_error"),
        ([{"chuong": "III", "dieu": [73, 46]}], "phạm vi ngược"),
        (
            [{"chuong": "III", "dieu": [74, 87]}, {"chuong": "III", "dieu": [46, 74]}],
            "chồng lấn",
        ),
    ],
)
def test_manifest_rejects_invalid_sections(document, sections, error):
    with pytest.raises(ValidationError, match=error):
        type(document).model_validate({**document.model_dump(), "cau_truc": sections})
