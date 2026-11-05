import unicodedata

import pytest

from ragdn.loai_hinh import detect_loai_hinh


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Công ty TNHH hai thành viên trở lên họp khi nào?", "tnhh_2tv"),
        ("Công ty trách nhiệm hữu hạn từ hai thành viên trở lên.", "tnhh_2tv"),
        ("Công ty TNHH 2 thành viên.", "tnhh_2tv"),
        ("TNHH 2TV", "tnhh_2tv"),
        ("tnhh2tv", "tnhh_2tv"),
        ("TNHH nhiều thành viên", "tnhh_2tv"),
        ("Công ty trách nhiệm hữu hạn nhiều thành viên", "tnhh_2tv"),
        ("Công ty trách nhiệm hữu hạn một thành viên.", "tnhh_mtv"),
        ("TNHH 1 thành viên", "tnhh_mtv"),
        ("TNHH MTV", "tnhh_mtv"),
        ("TNHH 1TV", "tnhh_mtv"),
        ("Doanh nghiệp nhà nước được tổ chức thế nào?", "dnnn"),
        ("DNNN", "dnnn"),
        ("Công ty cổ phần có quyền gì?", "ctcp"),
        ("CTCP", "ctcp"),
        ("Công ty hợp danh.", "hop_danh"),
        ("Doanh nghiệp tư nhân.", "dntn"),
        ("DNTN", "dntn"),
        ("Tập đoàn kinh tế.", "nhom_cty"),
        ("Nhóm công ty.", "nhom_cty"),
    ],
)
def test_detect_explicit_company_type(question, expected):
    assert detect_loai_hinh(question) == expected


@pytest.mark.parametrize("members", [3, 9, 10, 20, 100])
def test_detect_tnhh_with_multiple_members(members):
    assert detect_loai_hinh(f"tnhh {members} thanh vien") == "tnhh_2tv"
    assert detect_loai_hinh(f"Công ty trách nhiệm hữu hạn {members} thành viên") == "tnhh_2tv"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Đại hội đồng cổ đông họp khi nào?", "ctcp"),
        ("ĐHĐCĐ", "ctcp"),
        ("Hội đồng quản trị và HĐQT.", "ctcp"),
        ("Phát hành cổ phiếu.", "ctcp"),
        ("Thành viên hợp danh có quyền gì?", "hop_danh"),
        ("Cơ quan đại diện chủ sở hữu.", "dnnn"),
        ("Công ty mẹ và công ty con.", "nhom_cty"),
    ],
)
def test_detect_indirect_company_type(question, expected):
    assert detect_loai_hinh(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "Chuyển đổi TNHH MTV thành CTCP.",
        "TNHH MTV thành TNHH 2TV.",
        "Chuyển đổi TNHH MTV thành CTCP cần ĐHĐCĐ thông qua.",
        "Chuyển đổi TNHH MTV thành TNHH 2TV có được phát hành cổ phiếu không?",
        "So sánh doanh nghiệp nhà nước với công ty cổ phần.",
        "Thành viên hợp danh và hội đồng quản trị có quyền gì?",
        "Công ty mẹ và cơ quan đại diện chủ sở hữu.",
    ],
)
def test_multiple_company_types_are_ambiguous(question):
    assert detect_loai_hinh(question) is None


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Công ty TNHH hai thành viên muốn mua cổ phiếu.", "tnhh_2tv"),
        ("Công ty trách nhiệm hữu hạn 10 thành viên muốn mua cổ phiếu.", "tnhh_2tv"),
        ("Doanh nghiệp nhà nước có hội đồng quản trị.", "dnnn"),
        ("TNHH 2TV và công ty TNHH hai thành viên.", "tnhh_2tv"),
    ],
)
def test_explicit_company_type_takes_priority(question, expected):
    assert detect_loai_hinh(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "",
        "Thủ tục đăng ký doanh nghiệp.",
        "Công ty TNHH có Hội đồng thành viên.",
        "Doanh nghiệp có vốn nhà nước.",
        "Chủ sở hữu công ty.",
        "Tổng công ty.",
        "mactcp và xhdqt là mã nội bộ.",
        "tnhh 2tvabc",
        "tnhh 0 thanh vien",
    ],
)
def test_unknown_company_type_returns_none(question):
    assert detect_loai_hinh(question) is None


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("CONG TY TRACH NHIEM HUU HAN MOT THANH VIEN", "tnhh_mtv"),
        ("cong ty\nTNHH\thai thanh vien", "tnhh_2tv"),
        (unicodedata.normalize("NFD", "ĐẠI HỘI ĐỒNG CỔ ĐÔNG"), "ctcp"),
    ],
)
def test_detect_normalized_vietnamese(question, expected):
    assert detect_loai_hinh(question) == expected
