# src/ragdn/chunk.py
import collections
import re

RE_REF = re.compile(r"Điều\s+(\d+[a-zđ]?)\b")


def is_repealed(notes):
    return any(n["verb"] == "bị bãi bỏ" and not n["target"].startswith("Nội dung") for n in notes)


def chunk(articles, meta, max_chars=1800):
    chunks = []
    for a in articles:
        a_repealed = is_repealed(a["notes"]) or bool(a["khoans"]) and all(is_repealed(k["notes"]) for k in a["khoans"])
        header = f"{meta['ten_ngan']} | Chương {a['chuong'] or '-'} | Điều {a['dieu']}. {a['tieu_de']}"
        pieces = [(None, " ".join(a["intro"]), [])] if a["intro"] else []
        for k in a["khoans"]:
            txt = " ".join(k["lines"])
            if is_repealed(k["notes"]) and not a_repealed:
                txt = f"{k['k']}. (Khoản này đã bị bãi bỏ bởi {k['notes'][0]['source']})"
            pieces.append((k["k"], txt, k["notes"]))
        groups, cur, size = [], [], 0
        for pc in pieces:                          # gom khoản, không cắt giữa khoản
            if cur and size + len(pc[1]) > max_chars:
                groups.append(cur); cur, size = [], 0
            cur.append(pc); size += len(pc[1])
        if cur:
            groups.append(cur)
        for part, g in enumerate(groups):
            body = "\n".join(t for _, t, _ in g)
            notes = a["notes"] + [n for _, _, ns in g for n in ns]
            chunks.append({
                "chunk_id": f"{meta['doc_id']}:d{a['dieu']}:p{part}",
                "doc_id": meta["doc_id"], "ten_vb": meta["ten_ngan"], "vai_tro": meta["vai_tro"],
                "dieu": a["dieu"], "tieu_de": a["tieu_de"], "chuong": a["chuong"],
                "khoan": [k for k, _, _ in g if k], "page": a["page"],
                "hieu_luc": not a_repealed,
                "sua_doi": list(dict.fromkeys(f"{n['target']} {n['verb']} bởi {n['source']}" for n in notes)),
                "refs": sorted(set(RE_REF.findall(body)) - {a["dieu"]}),
                "ngay_hieu_luc": meta["ngay_hieu_luc"],
                "text": f"{header}\n{body}",
            })
    return chunks


def check_corpus(chunks, manifest_docs):
    """Trả về danh sách lỗi; rỗng mới được build index."""
    errors, by_doc = [], collections.defaultdict(set)
    for c in chunks:
        by_doc[c["doc_id"]].add(int(re.match(r"\d+", c["dieu"]).group()))
    for d in manifest_docs:
        nums = by_doc.get(d["doc_id"], set())
        if not nums:
            errors.append(f"{d['doc_id']}: không có chunk nào"); continue
        if miss := [n for n in range(1, max(nums) + 1) if n not in nums]:
            errors.append(f"{d['doc_id']}: thiếu Điều {miss}")
        if d.get("so_dieu") and max(nums) != d["so_dieu"]:
            errors.append(f"{d['doc_id']}: Điều cuối {max(nums)} ≠ so_dieu {d['so_dieu']}")
    ids = [c["chunk_id"] for c in chunks]
    if dup := [i for i, n in collections.Counter(ids).items() if n > 1]:
        errors.append(f"chunk_id trùng: {dup[:5]}")
    return errors