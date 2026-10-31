# tests/test_generate_fallback.py
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError

import ragdn.generate as gen


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


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))


