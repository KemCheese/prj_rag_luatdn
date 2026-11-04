"""T1 — Test parse.py: RE_MUC, field muc, guard _is_heading_tail, reset muc khi đổi Chương."""
from ragdn.parse import parse


def _lines(*texts):
    """Tạo danh sách (text, page) từ các chuỗi."""
    return [(t, i + 1) for i, t in enumerate(texts)]


# -----------------------------------------------------------------------
# 1. Luồng chính: Chương III → Mục 1 → Điều 46 → Mục 2 → Điều 74 →
#    Chương IV → Điều 88 → "Chương III hoặc công ty cổ phần…" → Điều 90
# -----------------------------------------------------------------------
def test_main_flow():
    lines = _lines(
        "Chương III",
        "Mục 1",
        "Điều 46. Công ty TNHH hai thành viên trở lên",
        "Nội dung điều 46.",
        "Mục 2",
        "Điều 74. Công ty TNHH một thành viên",
        "Nội dung điều 74.",
        "Chương IV",
        "Điều 88. Doanh nghiệp nhà nước",
        "Nội dung điều 88.",
        "Chương III hoặc công ty cổ phần theo quy định tại Chương V",
        "Điều 90. Quyền của cổ đông",
        "Nội dung điều 90.",
    )
    arts = parse(lines, "luat_dn_2020")
    lookup = {a["dieu"]: a for a in arts}

    assert lookup["46"]["chuong"] == "III"
    assert lookup["46"]["muc"] == "1"

    assert lookup["74"]["chuong"] == "III"
    assert lookup["74"]["muc"] == "2"

    # Chương IV: muc phải reset về None
    assert lookup["88"]["chuong"] == "IV"
    assert lookup["88"]["muc"] is None

    # "Chương III hoặc…" phải bị guard bỏ qua (có chữ thường) → vẫn IV
    assert lookup["90"]["chuong"] == "IV"
    assert lookup["90"]["muc"] is None


# -----------------------------------------------------------------------
# 2. "Mục 2 Chương này quy định…" nằm trong thân bài → không phải đề mục
# -----------------------------------------------------------------------
def test_muc_in_body_not_heading():
    lines = _lines(
        "Chương III",
        "Mục 1",
        "Điều 46. Tiêu đề",
        "Mục 2 Chương này quy định về thành viên.",
        "Mục 2",
        "Điều 74. Tiêu đề khác",
    )
    arts = parse(lines, "test")
    lookup = {a["dieu"]: a for a in arts}

    # "Mục 2 Chương này quy định…" có chữ thường → bị guard bỏ → nằm trong body Điều 46
    assert lookup["46"]["muc"] == "1"
    assert any("Mục 2 Chương này" in ln for ln in
               [l for k in lookup["46"]["khoans"] for l in k["lines"]]
               + lookup["46"]["intro"])
    assert lookup["74"]["muc"] == "2"


# -----------------------------------------------------------------------
# 3. "Mục 4 HÀNH VI VI PHẠM…" (chung dòng, viết hoa) → nhận là đề mục
# -----------------------------------------------------------------------
def test_muc_with_uppercase_title():
    lines = _lines(
        "Chương II",
        "Mục 1",
        "Điều 5. Vi phạm loại 1",
        "Nội dung.",
        "Mục 2",
        "Điều 8. Vi phạm loại 2",
        "Nội dung.",
        "Mục 3",
        "Điều 12. Vi phạm loại 3",
        "Nội dung.",
        "Mục 4 HÀNH VI VI PHẠM QUY ĐỊNH VỀ ĐĂNG KÝ DOANH NGHIỆP",
        "Điều 24. Tiêu đề",
        "Nội dung.",
    )
    arts = parse(lines, "nd_122")
    lookup = {a["dieu"]: a for a in arts}
    assert lookup["24"]["muc"] == "4"


# -----------------------------------------------------------------------
# 4. Hai lần "Chương V" → muc được đặt lại (guard >= cho phép)
# -----------------------------------------------------------------------
def test_two_chuong_v_resets_muc():
    lines = _lines(
        "Chương V",
        "Mục 1",
        "Điều 28. Tiêu đề A",
        "Nội dung.",
        "Mục 2",
        "Điều 30. Tiêu đề B",
        "Nội dung.",
        "Chương V",          # lần 2: guard >= cho phép, muc reset
        "Điều 32. Tiêu đề C",
        "Nội dung.",
    )
    arts = parse(lines, "nd_47")
    lookup = {a["dieu"]: a for a in arts}
    assert lookup["28"]["muc"] == "1"
    assert lookup["30"]["muc"] == "2"
    # Sau lần "Chương V" thứ 2, muc reset
    assert lookup["32"]["chuong"] == "V"
    assert lookup["32"]["muc"] is None


# -----------------------------------------------------------------------
# 5. track_quotes=True: "Mục 3" nằm trong ngoặc kép → bỏ qua
# -----------------------------------------------------------------------
def test_muc_inside_quotes_ignored():
    lines = _lines(
        "Chương I",
        "Mục 1",
        "Điều 1. Tiêu đề",
        "\u201cMục 3 QUY ĐỊNH MỚI\u201d",      # trong ngoặc kép
        "Nội dung tiếp.",
    )
    arts = parse(lines, "test", track_quotes=True)
    assert arts[0]["muc"] == "1"  # không bị đổi thành 3


# -----------------------------------------------------------------------
# 6. Không có Mục → muc = None cho tất cả
# -----------------------------------------------------------------------
def test_no_muc():
    lines = _lines(
        "Chương I",
        "Điều 1. Phạm vi điều chỉnh",
        "Nội dung.",
        "Chương II",
        "Điều 5. Đối tượng",
        "Nội dung.",
    )
    arts = parse(lines, "test")
    assert all(a["muc"] is None for a in arts)


# -----------------------------------------------------------------------
# 7. guard đơn điệu cho Mục: val > last_muc_val (nghiêm ngặt)
# -----------------------------------------------------------------------
def test_muc_monotonic_guard():
    lines = _lines(
        "Chương II",
        "Mục 1",
        "Điều 5. Tiêu đề",
        "Mục 1 VÀ CÁC QUY ĐỊNH KHÁC",  # muc=1 lại xuất hiện → val == last → không qua guard >
        "Điều 8. Tiêu đề khác",
    )
    arts = parse(lines, "test")
    lookup = {a["dieu"]: a for a in arts}
    # "Mục 1 VÀ CÁC QUY ĐỊNH KHÁC" lần 2 bị guard > chặn → nằm trong body
    assert lookup["5"]["muc"] == "1"
    assert lookup["8"]["muc"] == "1"  # vẫn dùng muc cũ

