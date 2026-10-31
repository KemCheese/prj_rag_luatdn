import re, unicodedata
import pymupdf

RE_CHUONG = re.compile(r"^Chương\s+([IVXLCDM]+)\b")
RE_DIEU   = re.compile(r"^Điều\s+(\d+)([a-zđ]?)\s*[.,:](?:\s+(?=[A-ZĐÀ-Ỹ])(.*))?$")
RE_KHOAN  = re.compile(r"^(\d+)([a-zđ]?)\.\s+\S")
RE_NOTE   = re.compile(
    r"^-?\s*(?P<target>(?:Điểm|Khoản|Điều|Chương|Mục|Nội dung này|Cụm từ)[^\n]*?)\s+"
    r"(?P<verb>được sửa đổi, bổ sung|được sửa đổi|được bổ sung|bị bãi bỏ|được thay thế|bị thay thế|được đính chính)"
    r"\s+bởi\s+(?P<source>.+?\d+/\d{4}/(?:QH\d+|NĐ-CP|TT-[A-ZĐ]+))\s*$")
RE_END    = re.compile(r"^(Nơi nhận:|TM\.\s*CHÍNH PHỦ|Luật này được Quốc hội|CHỦ TỊCH QUỐC HỘI)")

_SUFFIX_ORDER = "aăâbcdđeêfghijklmnoôơpqrstuưvwxyz"


def _order_key(num: str | int, suffix: str = "") -> tuple[int, int]:
    if not suffix:
        return (int(num), 0)
    idx = _SUFFIX_ORDER.find(suffix.lower())
    return (int(num), idx + 1 if idx != -1 else 999)


def read_pdf(path):
    lines = []
    for pno, page in enumerate(pymupdf.open(path), 1):
        for ln in page.get_text(sort=True).splitlines():
            t = unicodedata.normalize("NFC", ln).strip()
            if t:
                lines.append((t, pno))
    if not lines:
        raise ValueError(f"{path}: không có text layer — bản scan, cần OCR")
    return lines


def parse(lines, doc_id, track_quotes=False):
    articles, cur, chuong, last_dieu, depth = [], None, None, (0, 0), 0
    for t, p in lines:
        in_quote = track_quotes and (depth > 0 or t.startswith("“"))
        depth = max(0, depth + t.count("“") - t.count("”"))
        if RE_END.match(t):
            break
        if m := RE_NOTE.match(t):                
            if cur is not None:
                tgt = cur["khoans"][-1] if cur["khoans"] else cur
                if m.groupdict() not in tgt["notes"]:
                    tgt["notes"].append(m.groupdict())
            continue
        if m := RE_CHUONG.match(t):
            chuong = m.group(1)
            continue
        m = None if in_quote else RE_DIEU.match(t)
        if m and (dieu_k := _order_key(m.group(1), m.group(2))) > last_dieu:
            last_dieu = dieu_k
            cur = {"doc_id": doc_id, "dieu": m.group(1) + m.group(2),
                   "tieu_de": (m.group(3) or "").rstrip("."), "chuong": chuong,
                   "page": p, "intro": [], "khoans": [], "notes": []}
            articles.append(cur)
            continue
        if cur is None:
            continue                               
        m = None if in_quote else RE_KHOAN.match(t)
        if cur["khoans"]:
            km = re.match(r"^(\d+)([a-zđ]?)$", cur["khoans"][-1]["k"])
            last_k, last_k_key = int(km.group(1)), _order_key(km.group(1), km.group(2))
        else:
            last_k, last_k_key = 0, (0, 0)
        if m and last_k <= int(m.group(1)) <= last_k + 3 and _order_key(m.group(1), m.group(2)) > last_k_key:
            cur["khoans"].append({"k": m.group(1) + m.group(2), "lines": [t], "notes": []})
        elif cur["khoans"]:
            cur["khoans"][-1]["lines"].append(t)   
        elif not cur["tieu_de"]:
            cur["tieu_de"] = t.rstrip(".")         
        elif not cur["intro"] and not t[0].isupper() and len(cur["tieu_de"]) < 200:
            cur["tieu_de"] += " " + t              
        else:
            cur["intro"].append(t)
    return articles
