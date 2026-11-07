# tests/test_generate_fallback.py
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from types import MappingProxyType
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (str(ROOT), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import httpx
import pytest
from openai import APITimeoutError

import ragdn.generate as gen
import ragdn.pipeline as pipeline_module
from ragdn.pipeline import RAG
from ragdn.query_config import QueryConfig
from ragdn.query_models import (
    EvidenceRecord,
    IssueAssessment,
    IssueRecord,
    IssueScope,
    MissingInformation,
    ProcessingReport,
    QuestionPlan,
    SourceSpan,
)
from ragdn.query_pipeline import ContextView, QueryResult, ReaderContext, build_mapper_prompt


CHUNKS = [
    {
        "chunk_id": "luat_dn_2020:d4:p0",
        "doc_id": "luat_dn_2020",
        "ten_vb": "Luật Doanh nghiệp",
        "vai_tro": "chinh",
        "dieu": "4",
        "tieu_de": "Giải thích từ ngữ",
        "chuong": "I",
        "khoan": ["1", "5"],
        "page": 1,
        "hieu_luc": True,
        "sua_doi": [
            "Khoản 5 Điều 4 được sửa đổi, bổ sung bởi Điểm a Khoản 1 Điều 1 Luật số 76/2025/QH15"
        ],
        "refs": ["88"],
        "ngay_hieu_luc": "2021-01-01",
        "text": (
            "Luật Doanh nghiệp | Chương I | Điều 4. Giải thích từ ngữ\n"
            "1. Bản sao là giấy tờ được sao từ sổ gốc.\n"
            "5. Cổ đông sáng lập là cổ đông sở hữu ít nhất một cổ phần phổ thông và ký tên "
            "trong danh sách cổ đông sáng lập của công ty cổ phần; quy định liên quan tại Điều 88."
        ),
    },
    {
        "chunk_id": "luat_dn_2020:d206:p0",
        "doc_id": "luat_dn_2020",
        "ten_vb": "Luật Doanh nghiệp",
        "vai_tro": "chinh",
        "dieu": "206",
        "tieu_de": "Tạm ngừng, đình chỉ hoạt động, chấm dứt kinh doanh",
        "chuong": "IX",
        "khoan": ["1"],
        "page": 90,
        "hieu_luc": True,
        "sua_doi": [],
        "refs": [],
        "ngay_hieu_luc": "2021-01-01",
        "text": (
            "Luật Doanh nghiệp | Chương IX | Điều 206. Tạm ngừng kinh doanh\n"
            "1. Doanh nghiệp phải thông báo bằng văn bản cho Cơ quan đăng ký kinh doanh chậm nhất "
            "là 03 ngày làm việc trước ngày tạm ngừng kinh doanh. Thời hạn tạm ngừng kinh doanh "
            "mỗi lần không được quá 12 tháng và tổng thời gian tạm ngừng liên tiếp không quá 24 tháng."
        ),
    },
]


def _make_retriever(low_confidence: bool = False, flag: str | None = None):
    return SimpleNamespace(
        chunks=CHUNKS,
        retrieve=lambda q: {
            "hits": [] if low_confidence else [0, 1],
            "low_confidence": low_confidence,
            "flag": flag,
        },
    )


def _script_llm(monkeypatch, responses: list):
    calls = []

    def fake_call_llm(system: str, user: str, model: str | None = None, **kwargs):
        calls.append({"system": system, "user": user, "model": model, **kwargs})
        idx = len(calls) - 1
        item = responses[idx]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(gen, "call_llm", fake_call_llm)
    return calls


def test_retry_refusal_keeps_verified_first_answer(monkeypatch):
    """Lần 1 có 1 câu đúng [1] + 1 câu lỗi Điều 99; lần 2 LLM đầu hàng trả REFUSAL -> phải giữ câu đúng."""
    first = (
        "Theo khoản 5 Điều 4 Luật Doanh nghiệp, cổ đông sáng lập là cổ đông sở hữu ít nhất "
        "một cổ phần phổ thông [1]. Ngoài ra theo Điều 99 thì cổ đông phải nộp 500 triệu [1]."
    )
    calls = _script_llm(monkeypatch, [first, gen.REFUSAL])

    res = gen.generate_answer("Cổ đông sáng lập là gì?", _make_retriever())

    assert len(calls) == 2
    assert res["refused"] is False
    assert res.get("partial") is True
    assert "cổ đông sáng lập" in res["answer"]
    assert "Điều 99" not in res["answer"]
    assert gen.REFUSAL not in res["answer"]


def test_failed_retry_salvages_best_candidate(monkeypatch):
    """Cả 2 bản đều dính lỗi ở 1 câu -> chọn bản giữ được nhiều câu đúng có [n] hơn và kèm PRUNED_NOTE."""
    first = (
        "Theo Điều 4 Luật Doanh nghiệp, bản sao là giấy tờ được sao từ sổ gốc [1]. "
        "Theo Điều 99 phải phạt 999 triệu [1]."
    )
    second = (
        "Theo khoản 1 Điều 4 Luật Doanh nghiệp, bản sao là giấy tờ được sao từ sổ gốc [1]. "
        "Theo khoản 5 Điều 4 Luật Doanh nghiệp, cổ đông sáng lập ký tên trong danh sách [1]. "
        "Theo Điều 999 bị thu hồi giấy phép [1]."
    )
    _script_llm(monkeypatch, [first, second])

    res = gen.generate_answer("Giải thích từ ngữ Điều 4?", _make_retriever())

    assert res["refused"] is False
    assert res.get("partial") is True
    assert "khoản 1 Điều 4" in res["answer"]
    assert "khoản 5 Điều 4" in res["answer"]
    assert "Điều 999" not in res["answer"]
    assert gen.PRUNED_NOTE in res["answer"]


def test_refusal_substring_no_bypass(monkeypatch):
    """RC4: Câu trả lời vừa chứa REFUSAL vừa bịa Điều 99 [7] KHÔNG được lọt qua bộ lọc trích dẫn."""
    injected = f"{gen.REFUSAL} Tuy nhiên theo Điều 99 thì công ty vẫn được phép hoạt động [7]."
    clean_second = "Theo khoản 1 Điều 4 Luật Doanh nghiệp, bản sao là giấy tờ được sao từ sổ gốc [1]."
    calls = _script_llm(monkeypatch, [injected, clean_second])

    res = gen.generate_answer("Bản sao là gì?", _make_retriever())

    assert len(calls) == 2
    assert res["refused"] is False
    assert "Điều 99" not in res["answer"]
    assert "khoản 1 Điều 4" in res["answer"]


def test_amendment_note_dieu_allowed(monkeypatch):
    """RC1: Câu dẫn Điều 1 Luật số 76/2025/QH15 trong ghi chú sửa đổi của nguồn [1] phải hợp lệ ngay lần 1."""
    ans = (
        "Theo khoản 5 Điều 4 Luật Doanh nghiệp (được sửa đổi, bổ sung bởi Điểm a Khoản 1 Điều 1 "
        "Luật số 76/2025/QH15), cổ đông sáng lập sở hữu ít nhất một cổ phần phổ thông [1]."
    )
    calls = _script_llm(monkeypatch, [ans])

    res = gen.generate_answer("Khoản 5 Điều 4 được sửa đổi bởi văn bản nào?", _make_retriever())

    assert len(calls) == 1
    assert res["answer"] == ans
    assert res["refused"] is False
    assert res.get("partial") is False


def test_doc_year_allowed(monkeypatch):
    """RC6: Nhắc 'Luật Doanh nghiệp 2020' (năm của doc_id/so_hieu trong manifest) không bị coi là số bịa."""
    ans = (
        "Theo khoản 1 Điều 206 Luật Doanh nghiệp 2020, thời hạn tạm ngừng kinh doanh mỗi lần "
        "không được quá 12 tháng và tổng thời gian liên tiếp không quá 24 tháng [2]."
    )
    calls = _script_llm(monkeypatch, [ans])

    res = gen.generate_answer("Thời hạn tạm ngừng kinh doanh tối đa?", _make_retriever())

    assert len(calls) == 1
    assert res["answer"] == ans


def test_question_numbers_in_uncited_sentence(monkeypatch):
    """RC7: Câu không gắn [n] nhắc lại dữ kiện số trong câu hỏi tình huống (02 đợt, 500 triệu) là hợp lệ."""
    q = (
        "Công ty Bình Minh có vốn 500 triệu đồng đã tạm ngừng kinh doanh liên tiếp 02 đợt "
        "(mỗi đợt 12 tháng, tổng cộng 24 tháng). Công ty có được tạm ngừng đợt 3 không?"
    )
    ans = (
        "Trong tình huống này, công ty có vốn 500 triệu đồng và đã tạm ngừng liên tiếp 02 đợt. "
        "Theo khoản 1 Điều 206 Luật Doanh nghiệp 2020, thời hạn tạm ngừng mỗi lần không quá 12 tháng "
        "và tổng thời gian tạm ngừng liên tiếp không quá 24 tháng, nên công ty không được tạm ngừng liên tiếp thêm [2]."
    )
    calls = _script_llm(monkeypatch, [ans])

    res = gen.generate_answer(q, _make_retriever())

    assert len(calls) == 1
    assert res["answer"] == ans


def test_false_premise_number_still_blocked(monkeypatch):
    """Test bảo vệ: Nếu câu hỏi có tiền đề sai ('thời hạn 120 ngày') và LLM khẳng định '120 ngày [2]' -> phải bị chặn."""
    q = "Theo Điều 206, doanh nghiệp thông báo trước 120 ngày đúng không?"
    bad_first = "Đúng, theo khoản 1 Điều 206 Luật Doanh nghiệp, doanh nghiệp phải thông báo trước 120 ngày [2]."
    fixed_second = (
        "Không phải 120 ngày. Theo khoản 1 Điều 206 Luật Doanh nghiệp, doanh nghiệp phải thông báo "
        "chậm nhất là 03 ngày làm việc trước ngày tạm ngừng kinh doanh [2]."
    )
    calls = _script_llm(monkeypatch, [bad_first, fixed_second])

    res = gen.generate_answer(q, _make_retriever())

    assert len(calls) == 2
    assert "03 ngày làm việc" in res["answer"]


def test_uncited_invented_dieu_blocked(monkeypatch):
    """RC8a: Câu không có [n] nhưng tự bịa 'Theo Điều 999 Luật Doanh nghiệp' phải bị phát hiện và loại bỏ."""
    first = (
        "Theo Điều 999 Luật Doanh nghiệp, doanh nghiệp bị giải thể ngay lập tức. "
        "Theo khoản 1 Điều 206 Luật Doanh nghiệp, doanh nghiệp phải thông báo chậm nhất 03 ngày làm việc [2]."
    )
    _script_llm(monkeypatch, [first, first])

    res = gen.generate_answer("Tạm ngừng kinh doanh thông báo trước mấy ngày?", _make_retriever())

    assert "Điều 999" not in res["answer"]
    assert "03 ngày làm việc" in res["answer"]
    assert res.get("partial") is True


def test_foreign_document_and_precedent_blocked(monkeypatch):
    """RC8b: Viện dẫn văn bản ngoài kho ('Thông tư 96/2015/TT-BTC') hoặc 'Án lệ số 08/2016/AL' phải bị chặn."""
    first = (
        "Theo khoản 1 Điều 206 Luật Doanh nghiệp, doanh nghiệp thông báo chậm nhất 03 ngày làm việc [2]. "
        "Ngoài ra theo Thông tư 96/2015/TT-BTC và Án lệ số 08/2016/AL thì được miễn thuế [2]."
    )
    _script_llm(monkeypatch, [first, first])

    res = gen.generate_answer("Quy định tạm ngừng kinh doanh?", _make_retriever())

    assert "96/2015/TT-BTC" not in res["answer"]
    assert "Án lệ" not in res["answer"]
    assert "03 ngày làm việc" in res["answer"]
    assert res.get("partial") is True


def test_repair_call_sees_previous_answer(monkeypatch):
    """RC2: Lần gọi sửa (call 2) phải truyền bản trả lời lần 1 trong history với role='assistant'."""
    first = "Theo Điều 99 Luật Doanh nghiệp, bản sao là giấy tờ sao từ sổ gốc [1]."
    second = "Theo khoản 1 Điều 4 Luật Doanh nghiệp, bản sao là giấy tờ được sao từ sổ gốc [1]."
    calls = _script_llm(monkeypatch, [first, second])

    res = gen.generate_answer("Bản sao là gì?", _make_retriever())

    assert len(calls) == 2
    history = calls[1].get("history")
    assert history is not None and len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1] == {"role": "assistant", "content": first}
    assert res["answer"] == second


def test_repair_api_error_salvages_first(monkeypatch):
    """RC5: Lần gọi sửa bị APITimeoutError không làm nổ 503 mà cứu phần đúng của bản 1."""
    first = (
        "Theo khoản 1 Điều 4 Luật Doanh nghiệp, bản sao là giấy tờ được sao từ sổ gốc [1]. "
        "Theo Điều 99 phạt 500 triệu [1]."
    )
    timeout_err = APITimeoutError(request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"))
    _script_llm(monkeypatch, [first, timeout_err])

    res = gen.generate_answer("Bản sao là gì?", _make_retriever())

    assert res["refused"] is False
    assert res.get("partial") is True
    assert "sao từ sổ gốc" in res["answer"]
    assert "Điều 99" not in res["answer"]


def test_legit_refusal_when_not_substantive(monkeypatch):
    """Test bảo vệ: Khi bản 1 không có câu nào kiểm chứng được và bản 2 trả REFUSAL chuẩn -> refused=True."""
    first = "Tài liệu không đề cập đến thuế thu nhập doanh nghiệp."
    calls = _script_llm(monkeypatch, [first, gen.REFUSAL])

    res = gen.generate_answer("Mức thuế TNDN là bao nhiêu?", _make_retriever())

    assert len(calls) == 2
    assert gen.REFUSAL in calls[1]["user"]
    assert res["refused"] is True
    assert res["answer"] == gen.REFUSAL


def test_unverifiable_falls_back(monkeypatch):
    """Test bảo vệ: Cả 2 bản hoàn toàn sai và không cứu được câu nào -> trả UNVERIFIED, refused=False."""
    bad = "Theo Điều 999 phạt 888 tỷ đồng [9]."
    _script_llm(monkeypatch, [bad, bad])

    res = gen.generate_answer("Mức phạt?", _make_retriever())

    expected = getattr(
        gen, "UNVERIFIED", "Tôi chưa thể trả lời chắc chắn. Vui lòng xem các đoạn nguồn bên dưới."
    )
    assert res["refused"] is False
    assert res["answer"] == expected
    assert len(res["sources"]) == 2


def test_clean_first_single_call(monkeypatch):
    """Test bảo vệ: Bản 1 sạch hoàn toàn -> chỉ gọi LLM 1 lần."""
    ans = "Theo khoản 1 Điều 4 Luật Doanh nghiệp, bản sao là giấy tờ được sao từ sổ gốc [1]."
    calls = _script_llm(monkeypatch, [ans])

    res = gen.generate_answer("Bản sao là gì?", _make_retriever())

    assert len(calls) == 1
    assert res["answer"] == ans
    assert res["refused"] is False


def test_low_confidence_no_llm(monkeypatch):
    """Test bảo vệ: Khi retriever báo low_confidence -> trả REFUSAL ngay, không gọi LLM."""
    calls = _script_llm(monkeypatch, [])

    res = gen.generate_answer("Câu hỏi ngoài phạm vi?", _make_retriever(low_confidence=True))

    assert len(calls) == 0
    assert res["refused"] is True
    assert res["answer"] == gen.REFUSAL
    assert res["sources"] == []


def test_salvage_layout():
    """Hàm salvage bỏ đúng dòng gạch đầu dòng bị lỗi và giữ nguyên xuống dòng của các dòng đúng."""
    assert hasattr(gen, "salvage"), "Chưa có hàm salvage"
    text = (
        "Các quy định tại Điều 4:\n"
        "- Khoản 1 Điều 4: Bản sao là giấy tờ được sao từ sổ gốc [1].\n"
        "- Khoản 99 Điều 99: Phạt 500 triệu đồng [1].\n"
        "- Khoản 5 Điều 4: Cổ đông sáng lập sở hữu ít nhất một cổ phần phổ thông [1]."
    )
    cleaned, n_cited = gen.salvage(text, CHUNKS, "Điều 4 quy định gì?")
    assert n_cited == 2
    assert "Khoản 1 Điều 4" in cleaned
    assert "Khoản 5 Điều 4" in cleaned
    assert "Điều 99" not in cleaned
    assert "\n- Khoản 5 Điều 4" in cleaned


def test_is_refusal_exact():
    """Hàm is_refusal chỉ khớp toàn văn câu REFUSAL (bỏ qua hoa/thường và dấu câu), từ chối chuỗi con."""
    assert hasattr(gen, "is_refusal"), "Chưa có hàm is_refusal"
    assert gen.is_refusal(gen.REFUSAL) is True
    assert gen.is_refusal("  tôi không tìm thấy thông tin này trong các văn bản đã nạp! ") is True
    assert gen.is_refusal(f"{gen.REFUSAL} Tuy nhiên theo Điều 4 [1]...") is False
    assert gen.is_refusal("Câu trả lời bình thường [1].") is False


def test_reviewer_edge_cases_inline_cite_uppercase_dieu_and_number_isolation():
    """Kiểm tra các edge case từ code-reviewer & security-reviewer:
    1) Tách câu inline kiểu '. [1] Câu tiếp theo'
    2) Bắt 'ĐIỀU 999' viết hoa toàn bộ
    3) Bắt số đơn lẻ ('7 năm', '59 triệu') và chống lấy râu ông nọ cắm cằm bà kia giữa nguồn [1] và [2]."""
    inline = (
        "Theo khoản 1 Điều 4 Luật Doanh nghiệp, bản sao là giấy tờ được sao từ sổ gốc. [1] "
        "Theo ĐIỀU 999 thì bị phạt 7 năm hoặc 59 triệu [1]. "
        "Ngoài ra theo Điều 4 thì được tạm ngừng 24 tháng [1]."
    )
    cleaned, n_cited = gen.salvage(inline, CHUNKS, "Bản sao là gì?")
    assert n_cited == 1
    assert "sao từ sổ gốc. [1]" in cleaned
    assert "ĐIỀU 999" not in cleaned
    assert "24 tháng" not in cleaned


def test_outline_numbering_and_multi_khoan_not_flagged(monkeypatch):
    """RC9 & RC10: Số thứ tự đề mục (1., 2.1, 3.1.1, (1)..(9)) và cụm (khoản 1, 5) không bị phạt oan,
    không bị cắt rời dấu chấm đầu dòng, nhưng số tiền hàng nghìn bịa đặt ('500.000') vẫn bị bắt."""
    structured_ans = (
        "1. Vấn đề pháp lý: Xác định quy định giải thích từ ngữ và tạm ngừng kinh doanh.\n"
        "2. Quy định áp dụng:\n"
        "2.1 [1] Luật Doanh nghiệp, Điều 4 (khoản 1, 5): bản sao là giấy tờ được sao từ sổ gốc.\n"
        "2.2. [2] Luật Doanh nghiệp, Điều 206 (khoản 1): thông báo chậm nhất 03 ngày làm việc.\n"
        "3.1.1 Theo [1] Luật Doanh nghiệp, Điều 4 (khoản 1, 5), cổ đông sáng lập sở hữu ít nhất một cổ phần phổ thông.\n"
        "4.1 Kết luận: (1) Bản sao lấy từ sổ gốc [1]; (2) Tạm ngừng tối đa 12 tháng mỗi lần [2]."
    )
    calls = _script_llm(monkeypatch, [structured_ans])

    res = gen.generate_answer("Giải thích Điều 4 và Điều 206?", _make_retriever())

    assert len(calls) == 1
    assert res["refused"] is False
    assert res.get("partial") is False
    assert res["answer"] == structured_ans

    # Đảm bảo số tiền phân cách hàng nghìn bịa đặt ('500.000') vẫn bị bắt lỗi
    bad_thousand = "2.1 Theo khoản 1 Điều 4 Luật Doanh nghiệp, mức lệ phí là 500.000 đồng [1]."
    assert any("500.000" in e for e in gen.find_issues(bad_thousand, CHUNKS, "Lệ phí?"))


def test_cited_first_attempt_all_sentences_flawed_never_capitulates_to_refusal(monkeypatch):
    """RC11: Khi Bản 1 đã có trích dẫn [n] hợp lệ nhưng mọi câu đều vướng lỗi số,
    Lần 2 không được gợi ý REFUSAL và nếu LLM vẫn trả REFUSAL thì kết quả cuối phải là UNVERIFIED (refused=False)."""
    first = (
        "2.1 Theo khoản 1 Điều 4 Luật Doanh nghiệp, phạt 999 triệu đồng [1].\n"
        "2.2 Theo khoản 1 Điều 206 Luật Doanh nghiệp, phạt 888 triệu đồng [2]."
    )
    calls = _script_llm(monkeypatch, [first, gen.REFUSAL])

    res = gen.generate_answer("Mức phạt là bao nhiêu?", _make_retriever())

    assert len(calls) == 2
    assert gen.REFUSAL not in calls[1]["user"]
    assert res["refused"] is False
    assert res["answer"] == gen.UNVERIFIED
    assert len(res["sources"]) == 2


def test_rag_retrieve_and_answer_share_single_request_local_orchestration(monkeypatch):
    retrieve_calls = []

    def _retrieve(q):
        retrieve_calls.append(q)
        return {"hits": [0], "max_score": 0.92, "low_confidence": False, "flag": None, "loai_hinh": None}

    dummy = SimpleNamespace(chunks=[dict(CHUNKS[0])], retrieve=_retrieve)
    rag = RAG(manifest={"documents": []}, retriever=dummy)
    _script_llm(monkeypatch, ["Theo khoản 1 Điều 4 Luật Doanh nghiệp, bản sao là giấy tờ được sao từ sổ gốc [1]."] * 8)

    ret = rag.retrieve("Bản sao là gì?", mode="legacy")
    assert ret["hits"] == [0]
    assert ret["processing"]["mode"] == "legacy"
    assert len(ret["context_views"]) == 1
    assert len(ret["context_spans"]) == 1

    retrieve_calls.clear()
    ans = rag.answer("Bản sao là gì?", mode="legacy")
    assert retrieve_calls == ["Bản sao là gì?"]
    assert ans["refused"] is False
    assert ans["partial"] is False
    assert ans["partial_reasons"] == []
    assert ans["processing"]["mode"] == "legacy"

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(rag.answer, "Câu hỏi A?", mode="legacy")
        f2 = pool.submit(rag.answer, "Câu hỏi B?", variant="A")
        r1, r2 = f1.result(), f2.result()
    assert r1["processing"]["trace_id"] != r2["processing"]["trace_id"]


def test_context_view_synchronizes_reader_and_validator_while_returning_unmutated_parent_sources(monkeypatch):
    parent = dict(CHUNKS[0])
    clause1_text = "Luật Doanh nghiệp | Chương I | Điều 4. Giải thích từ ngữ\n1. Bản sao là giấy tờ được sao từ sổ gốc."
    start = parent["text"].index(clause1_text)
    end = start + len(clause1_text)
    view = ContextView(
        parent_chunk_id=parent["chunk_id"],
        citation_index=1,
        spans=(SourceSpan(start=start, end=end, quote=clause1_text),),
        text=clause1_text,
        whole_parent=False,
    )
    qres = QueryResult(
        original_question="Bản sao là gì?",
        retrieval={"hits": [0], "max_score": 0.95, "low_confidence": False, "flag": "query_pipeline", "loai_hinh": None},
        sources=(parent,),
        context_views=(view,),
        plan=None,
        evidence=(),
        processing=ProcessingReport(mode="direct", route_reason="direct_simple", status="not_assessed"),
        diagnostics={},
    )
    # Bản 1 dùng số 20 và Điều 88 chỉ có ở Khoản 5 (ngoài ContextView Khoản 1) -> phải bị bắt lỗi, sửa ở Bản 2
    bad_first = "Theo khoản 1 Điều 4 Luật Doanh nghiệp, bản sao chiếm 20 phần trăm theo Điều 88 [1]."
    good_second = "Theo khoản 1 Điều 4 Luật Doanh nghiệp, bản sao là giấy tờ được sao từ sổ gốc [1]."
    calls = _script_llm(monkeypatch, [bad_first, good_second])

    res = gen.generate_answer("Bản sao là gì?", _make_retriever(), query_result=qres)
    assert len(calls) == 2
    assert "20 phần trăm" not in calls[0]["user"]
    assert res["answer"] == good_second
    # Nguồn trả về cho người dùng vẫn giữ nguyên văn bản parent gốc chưa bị cắt
    assert res["sources"][0]["text"] == CHUNKS[0]["text"]
    assert res["sources"][0]["khoan"] == ["1", "5"]


def test_enhanced_mode_gates_reader_to_supported_and_conditional_issues_and_handles_no_supported_issue(monkeypatch):
    p0 = dict(CHUNKS[0])
    p1 = dict(CHUNKS[1])
    plan = QuestionPlan(
        original_question="Bản sao và tạm ngừng kinh doanh?",
        facts=[],
        issues=[
            IssueRecord(issue_id="i1", title="Khái niệm bản sao", main_query="bản sao là gì", scope=IssueScope()),
            IssueRecord(issue_id="i2", title="Tạm ngừng kinh doanh", main_query="tạm ngừng kinh doanh", scope=IssueScope()),
        ],
    )
    # Trường hợp 1: không có issue nào supported/conditional -> trả INSUFFICIENT_BASIS, không gọi call_llm
    no_sup_report = ProcessingReport(
        mode="enhanced",
        route_reason="multi_issue_question",
        status="insufficient",
        issues=[
            IssueAssessment(issue_id="i1", status="insufficient"),
            IssueAssessment(
                issue_id="i2",
                status="ambiguous",
                missing_information=[MissingInformation(description="Thiếu loại hình doanh nghiệp", scope_unresolved=True)],
            ),
        ],
        missing_information=[MissingInformation(description="Thiếu loại hình doanh nghiệp", scope_unresolved=True)],
        partial_reasons=["evidence_missing", "ambiguous_scope"],
    )
    qres_none = QueryResult(
        original_question="Bản sao và tạm ngừng kinh doanh?",
        retrieval={"hits": [0, 1], "max_score": 0.85, "low_confidence": False, "flag": "query_pipeline", "loai_hinh": None},
        sources=(p0, p1),
        context_views=(),
        plan=plan,
        evidence=(),
        processing=no_sup_report,
        diagnostics={},
    )
    calls = _script_llm(monkeypatch, ["Không được gọi"])
    res_none = gen.generate_answer("Bản sao và tạm ngừng kinh doanh?", _make_retriever(), query_result=qres_none)
    assert len(calls) == 0
    assert gen.INSUFFICIENT_BASIS in res_none["answer"]
    assert "Thiếu loại hình doanh nghiệp" in res_none["answer"]
    assert res_none["refused"] is False
    assert res_none["partial"] is False
    assert set(res_none["partial_reasons"]) == {"evidence_missing", "ambiguous_scope"}

    # Trường hợp 2: i1 supported (trên p0), i2 insufficient (trên p1) -> reader chỉ nhận p0 và <pham_vi_tra_loi>
    ev_i1 = EvidenceRecord(
        evidence_id="e1",
        issue_id="i1",
        parent_chunk_id=p0["chunk_id"],
        doc_id=p0["doc_id"],
        dieu=p0["dieu"],
        source_spans=[SourceSpan(start=0, end=len(p0["text"]), quote=p0["text"])],
    )
    ev_i2 = EvidenceRecord(
        evidence_id="e2",
        issue_id="i2",
        parent_chunk_id=p1["chunk_id"],
        doc_id=p1["doc_id"],
        dieu=p1["dieu"],
        source_spans=[SourceSpan(start=0, end=len(p1["text"]), quote=p1["text"])],
    )
    part_report = ProcessingReport(
        mode="enhanced",
        route_reason="multi_issue_question",
        status="partial",
        issues=[
            IssueAssessment(issue_id="i1", status="supported", evidence_ids=["e1"]),
            IssueAssessment(issue_id="i2", status="insufficient"),
        ],
        partial_reasons=["evidence_missing"],
    )
    qres_part = QueryResult(
        original_question="Bản sao và tạm ngừng kinh doanh?",
        retrieval={"hits": [0, 1], "max_score": 0.85, "low_confidence": False, "flag": "query_pipeline", "loai_hinh": None},
        sources=(p0, p1),
        context_views=(),
        plan=plan,
        evidence=(ev_i1, ev_i2),
        processing=part_report,
        diagnostics={},
    )
    calls = _script_llm(monkeypatch, ["Theo khoản 1 Điều 4 Luật Doanh nghiệp, bản sao là giấy tờ được sao từ sổ gốc [1]."])
    res_part = gen.generate_answer("Bản sao và tạm ngừng kinh doanh?", _make_retriever(), query_result=qres_part)
    assert len(calls) == 1
    assert "Điều 206" not in calls[0]["user"]
    assert "<pham_vi_tra_loi>" in calls[0]["user"]
    assert res_part["partial"] is True
    assert res_part["partial_reasons"] == ["evidence_missing"]
    assert len(res_part["sources"]) == 1 and res_part["sources"][0]["chunk_id"] == p0["chunk_id"]


def test_full_generation_and_repair_prompts_checked_against_reader_context_and_logging_is_content_free(monkeypatch, caplog):
    # 1. Repair context budget exhausted -> bỏ qua gọi LLM lần 2, cứu bản 1 và ghi cảnh báo
    first_mixed = (
        "Theo khoản 1 Điều 4 Luật Doanh nghiệp, bản sao là giấy tờ được sao từ sổ gốc [1]. "
        "Thời hạn giải quyết là 999 ngày làm việc [1]."
    )
    calls = _script_llm(monkeypatch, [first_mixed, "Không được gọi lần 2"])
    # Chọn context_tokens đủ cho lần 1 (~850 từ + reserve 2512) nhưng không đủ cho (user + first + repair)
    ctx = ReaderContext(context_tokens=3420, token_counter=lambda s: len(s.split()), tokenizer_verified=True)
    cfg = QueryConfig(context_tokens=3420)
    with caplog.at_level("WARNING"):
        res = gen.generate_answer("Bản sao là gì?", _make_retriever(), config=cfg, context=ctx)
    assert len(calls) == 1
    assert res["partial"] is True
    assert "generation_repair_context_exhausted" in res["processing"]["warnings"]
    assert "999 ngày" not in caplog.text

    # 2. Repair APIError không log nội dung thô từ upstream exception
    caplog.clear()
    _script_llm(
        monkeypatch,
        [
            first_mixed,
            APITimeoutError(request=httpx.Request("POST", "http://secret-internal-host.local/v1/chat?token=TOPSECRET")),
        ],
    )
    with caplog.at_level("WARNING"):
        res2 = gen.generate_answer("Bản sao là gì?", _make_retriever())
    assert res2["partial"] is True
    assert "TOPSECRET" not in caplog.text
    assert "secret-internal-host" not in caplog.text


def _whole_view(parent, index=1):
    return ContextView(parent["chunk_id"], index,
                       (SourceSpan(start=0, end=len(parent["text"]), quote=parent["text"]),), parent["text"], True)


def test_rag_answer_keeps_filtered_second_parent_views_metadata_and_repair_citations(monkeypatch):
    parents = deepcopy(CHUNKS)
    parents[1]["extra"] = MappingProxyType({"nested": ("kept",)})
    frozen = tuple(MappingProxyType(parent) for parent in parents)
    evidence = EvidenceRecord(evidence_id="e2", issue_id="i2", parent_chunk_id=parents[1]["chunk_id"],
                              doc_id=parents[1]["doc_id"], dieu="206", source_spans=list(_whole_view(parents[1]).spans))
    plan = QuestionPlan(original_question="Khái niệm và thời hạn?", facts=[], issues=[
        IssueRecord(issue_id="i1", title="Khái niệm", main_query="khái niệm"),
        IssueRecord(issue_id="i2", title="Thời hạn", main_query="thời hạn"),
    ])
    qres = QueryResult(plan.original_question, {"hits": [0, 1], "low_confidence": False}, frozen,
                       tuple(_whole_view(parent, index) for index, parent in enumerate(parents, 1)), plan, (evidence,),
                       ProcessingReport(mode="enhanced", route_reason="multi_issue", status="partial", issues=[
                           IssueAssessment(issue_id="i1", status="insufficient"),
                           IssueAssessment(issue_id="i2", status="supported", evidence_ids=["e2"]),
                       ], partial_reasons=["evidence_missing"]), {"variant": "D"})
    orchestrate = Mock(return_value=qres)
    monkeypatch.setattr(pipeline_module, "run_query_pipeline", orchestrate)
    retriever = SimpleNamespace(chunks=parents, candidate_batch=Mock(), retrieve=Mock(side_effect=AssertionError("double retrieval")))
    ctx = ReaderContext(100000)
    rag = RAG({"documents": []}, retriever, reader_context=ctx)
    first = "Theo Điều 4 Luật Doanh nghiệp, bản sao lấy từ sổ gốc [1]."
    second = "Theo khoản 1 Điều 206 Luật Doanh nghiệp, doanh nghiệp phải thông báo trước 03 ngày làm việc [1]."
    calls = _script_llm(monkeypatch, [first, second])
    original_validator = gen.find_issues
    validated_views = []

    def validate(answer, sources, *args):
        validated_views.append([(source["chunk_id"], source["text"]) for source in sources])
        return original_validator(answer, sources, *args)

    monkeypatch.setattr(gen, "find_issues", validate)
    result = rag.answer(plan.original_question)
    orchestrate.assert_called_once()
    retriever.retrieve.assert_not_called()
    assert len(calls) == 2 and result["answer"] == second
    assert "Điều 4." not in calls[0]["user"] and "Điều 206." in calls[0]["user"]
    assert calls[1]["history"][0]["content"] == calls[0]["user"]
    assert all(sources == [(parents[1]["chunk_id"], parents[1]["text"])] for sources in validated_views)
    assert result["sources"][0]["extra"] == {"nested": ["kept"]}
    proc = result["processing"]
    assert proc["context_views"][0]["parent_chunk_id"] == parents[1]["chunk_id"]
    assert proc["context_views"][0]["citation_index"] == 1 and len(proc["context_views"]) == 1
    assert {span["parent_chunk_id"] for span in proc["context_spans"]} == {parents[1]["chunk_id"]}
    assert proc["issues"][1]["title"] == "Thời hạn"
    metrics = proc["reader_context"]
    assert metrics["reader_calls"] == 2 and metrics["context_tokens"] == 100000
    assert metrics["method"] == "conservative_utf8_bytes"
    assert metrics["prompt_tokens"]["reader"] == sum(ctx.count(calls[0][key]) for key in ("system", "user"))
    assert metrics["prompt_tokens"]["repair"] == sum(ctx.count(text) for text in
        (calls[1]["system"], calls[0]["user"], first, calls[1]["user"]))
    assert result["diagnostics"] == {"variant": "D"}


def test_generation_context_overflow_never_retrieves_or_claims_a_legacy_result(monkeypatch):
    large = deepcopy(CHUNKS[0])
    large["text"] = "Nội dung rất dài. " * 5000
    small = deepcopy(CHUNKS[1])
    retriever = SimpleNamespace(chunks=[large, small], retrieve=Mock(side_effect=AssertionError("generation must not retrieve")))
    qres = QueryResult("Thời hạn?", {"hits": [0], "low_confidence": False}, (large,), (_whole_view(large),), None, (),
                       ProcessingReport(mode="direct", route_reason="single_explicit_lookup"), {})
    calls = _script_llm(monkeypatch, [])
    ctx = ReaderContext(4000, token_counter=lambda text: len(text.split()), tokenizer_verified=True)
    result = gen.generate_answer("Thời hạn?", retriever, query_result=qres, context=ctx, config=QueryConfig(context_tokens=4000))
    retriever.retrieve.assert_not_called()
    assert calls == [] and result["processing"]["mode"] == "direct"
    assert result["processing"]["status"] == "not_assessed" and result["partial"] is False
    assert result["processing"]["evidence"] == []
    assert result["processing"]["context_spans"] == [] and result["processing"]["context_views"] == []
    assert "context_budget_exhausted" in result["processing"]["warnings"]
    assert result["sources"] == [large] and large["text"] == "Nội dung rất dài. " * 5000
    assert result["processing"]["reader_context"]["reader_calls"] == 0
    assert result["answer"] == gen.UNVERIFIED


def test_enhanced_mapper_fits_but_complete_reader_prompt_overflows_without_retrieval(monkeypatch):
    parent = deepcopy(CHUNKS[0])
    question = "Bản sao?"
    plan = QuestionPlan(original_question=question, facts=[], issues=[IssueRecord(issue_id="i1", title="Bản sao", main_query="bản sao")])
    evidence = EvidenceRecord(evidence_id="e1", issue_id="i1", parent_chunk_id=parent["chunk_id"], doc_id=parent["doc_id"],
                              dieu="4", source_spans=list(_whole_view(parent).spans))
    qres = QueryResult(question, {"hits": [0], "low_confidence": False}, (parent,), (_whole_view(parent),), plan, (evidence,),
                       ProcessingReport(mode="enhanced", route_reason="planned", status="supported",
                                        issues=[IssueAssessment(issue_id="i1", status="supported", evidence_ids=["e1"])]), {})
    mapper_system, mapper_user = build_mapper_prompt(plan, [{"group_id": "g1", "parent_chunk_id": parent["chunk_id"],
                                                          "units": [{"unit_id": "u1", "text": parent["text"]}]}])
    counter = lambda text: len(text.split())
    limit = counter(mapper_system) + counter(mapper_user) + 2000 + 512
    ctx = ReaderContext(limit, token_counter=counter, tokenizer_verified=True)
    cfg = QueryConfig(context_tokens=limit)
    ctx.check(mapper_system, mapper_user, cfg)
    retriever = SimpleNamespace(chunks=[parent], retrieve=Mock(side_effect=AssertionError("second retrieval")))
    calls = _script_llm(monkeypatch, [])
    result = gen.generate_answer(question, retriever, query_result=qres, context=ctx, config=cfg)
    retriever.retrieve.assert_not_called()
    assert calls == [] and result["answer"] == gen.UNVERIFIED
    proc = result["processing"]
    assert proc["mode"] == "enhanced" and proc["status"] == "not_assessed"
    assert proc["route_reason"] == "reader_context_budget_exhausted"
    assert proc["issues"][0]["status"] == "not_processed" and proc["issues"][0]["evidence_ids"] == []
    assert proc["context_views"] == [] and proc["context_spans"] == [] and proc["evidence"] == []
    assert proc["budget"]["not_processed"] == ["i1"]
    assert proc["reader_context"]["reader_calls"] == 0 and proc["reader_context"]["prompt_tokens"] == {}
    assert result["sources"] == [parent] and result["partial"] is False


def test_unknown_context_fallback_keeps_one_legacy_retrieval_and_not_assessed(monkeypatch):
    retriever = SimpleNamespace(chunks=deepcopy(CHUNKS),
                                candidate_batch=Mock(side_effect=AssertionError("must not query without context")),
                                retrieve=Mock(return_value={"hits": [0], "low_confidence": False}))
    calls = _script_llm(monkeypatch, [])
    result = RAG({"documents": []}, retriever, reader_context=ReaderContext()).answer("Bản sao là gì?")
    retriever.retrieve.assert_called_once_with("Bản sao là gì?")
    retriever.candidate_batch.assert_not_called()
    assert result["processing"]["mode"] == "baseline_fallback"
    assert result["processing"]["status"] == "not_assessed"
    assert calls == [] and result["answer"] == gen.PROCESSING_FAILURE and result["refused"] is False
    assert result["processing"]["reader_context"]["reader_calls"] == 0
    assert "context_budget_unknown" in result["processing"]["warnings"]


def test_partial_view_never_imports_parent_only_notes_or_reference_metadata():
    parent = deepcopy(CHUNKS[0])
    parent["refs"].append("999")
    quote = parent["text"].split("\n5.")[0]
    view = ContextView(parent["chunk_id"], 1, (SourceSpan(start=0, end=len(quote), quote=quote),), quote, False)
    selected = gen.build_view_chunk(parent, view)
    assert selected["sua_doi"] == [] and selected["refs"] == [] and selected["khoan"] == ["1"]
    assert "76/2025" not in gen.build_context([selected])
    assert gen.find_issues("Năm 2025, áp dụng Điều 999 [1].", [selected], "Nội dung?")
    assert parent["sua_doi"] == CHUNKS[0]["sua_doi"] and parent["refs"] == ["88", "999"]
    whole = gen.build_view_chunk(parent, _whole_view(parent))
    assert "999" not in whole["refs"] and "88" in whole["refs"]


@pytest.mark.parametrize("bad_view", [
    ContextView("wrong-id", 1, (SourceSpan(start=0, end=1, quote="x"),), "x", False),
    ContextView(CHUNKS[0]["chunk_id"], 1, (SourceSpan(start=0, end=1, quote="x"),), "x", False),
])
def test_invalid_context_identity_or_span_propagates_without_reader(monkeypatch, bad_view):
    calls = _script_llm(monkeypatch, [])
    with pytest.raises(ValueError, match="reader_context_view"):
        gen.generate_answer("Bản sao?", _make_retriever(), retrieval={"hits": [0], "low_confidence": False}, context_views=[bad_view])
    assert calls == []


def test_supported_claim_without_evidence_is_not_sent_to_reader(monkeypatch):
    parent = deepcopy(CHUNKS[0])
    qres = QueryResult("Bản sao?", {"hits": [0], "low_confidence": False}, (parent,), (_whole_view(parent),), None, (),
                       ProcessingReport(mode="enhanced", route_reason="planned", status="supported",
                                        issues=[IssueAssessment(issue_id="i1", status="supported", evidence_ids=["absent"])]), {})
    calls = _script_llm(monkeypatch, [])
    result = gen.generate_answer("Bản sao?", _make_retriever(), query_result=qres)
    assert calls == [] and gen.INSUFFICIENT_BASIS in result["answer"]
    assert result["processing"]["status"] == "insufficient"
    assert result["processing"]["issues"][0]["status"] == "uncertain"


def test_generation_pruning_changes_supported_processing_to_partial(monkeypatch):
    parent = deepcopy(CHUNKS[0])
    evidence = EvidenceRecord(evidence_id="e1", issue_id="i1", parent_chunk_id=parent["chunk_id"], doc_id=parent["doc_id"],
                              dieu="4", source_spans=list(_whole_view(parent).spans))
    qres = QueryResult("Bản sao?", {"hits": [0], "low_confidence": False}, (parent,), (_whole_view(parent),), None, (evidence,),
                       ProcessingReport(mode="enhanced", route_reason="planned", status="supported",
                                        issues=[IssueAssessment(issue_id="i1", status="supported", evidence_ids=["e1"])]), {})
    answer = "Theo Điều 4 Luật Doanh nghiệp, bản sao là giấy tờ lấy từ sổ gốc [1]. Theo Điều 999 phạt 888 triệu [1]."
    _script_llm(monkeypatch, [answer, answer])
    result = gen.generate_answer("Bản sao?", _make_retriever(), query_result=qres)
    assert result["partial"] is True and result["processing"]["status"] == "partial"
    assert "generation_pruned" in result["partial_reasons"] and "generation_pruned" in result["processing"]["partial_reasons"]


def main() -> int:
    return int(pytest.main([__file__, *sys.argv[1:]]))


if __name__ == "__main__":
    raise SystemExit(main())


