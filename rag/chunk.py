"""แบ่งเอกสารเป็น chunk พร้อม metadata (เวิร์กช็อป 1)

แต่ละ chunk เป็น dict ที่มีอย่างน้อย:
chunk_id ("<doc_id>#<ลำดับ>"), doc_id, title, section, status, effective_date, text
"""
from rag.ingest import load_docs

MAX_CHARS = 600


def split_sections(body: str) -> list[tuple[str, list[str]]]:
    """คืน [(ชื่อหัวข้อ, บรรทัดเนื้อหา)] ตามหัวข้อ ## ข้ามหัวเรื่อง # และบรรทัดว่าง"""
    sections, current = [], None
    for line in body.splitlines():
        if line.startswith("## "):
            current = (line[3:].strip(), [])
            sections.append(current)
        elif line.strip() and not line.startswith("# "):
            if current is None:  # เนื้อหาก่อนหัวข้อ ## แรก
                current = ("", [])
                sections.append(current)
            current[1].append(line.strip())
    return sections


def pack_lines(lines: list[str], prefix: str, max_chars: int) -> list[str]:
    """รวมบรรทัดเป็นก้อนที่ยาวไม่เกิน max_chars (นับรวม prefix) บรรทัดที่ยาวเกินจะถูกตัดตามตัวอักษร"""
    budget = max_chars - len(prefix)
    pieces = []
    for line in lines:
        pieces += [line[i:i + budget] for i in range(0, len(line), budget)]
    texts, buf = [], ""
    for p in pieces:
        if buf and len(buf) + 1 + len(p) > budget:
            texts.append(prefix + buf)
            buf = ""
        buf = f"{buf}\n{p}" if buf else p
    if buf:
        texts.append(prefix + buf)
    return texts


def chunk_docs(docs: list[dict] | None = None, max_chars: int = MAX_CHARS) -> list[dict]:
    docs = load_docs() if docs is None else docs
    chunks = []
    for doc in docs:
        meta = {k: v for k, v in doc.items() if k != "body"}
        n = 0
        for section, lines in split_sections(doc["body"]):
            prefix = f"{doc['title']} · {section}\n" if section else f"{doc['title']}\n"
            for text in pack_lines(lines, prefix, max_chars):
                n += 1
                chunks.append({**meta, "chunk_id": f"{doc['doc_id']}#{n}", "section": section, "text": text})
    return chunks


if __name__ == "__main__":
    cs = chunk_docs()
    print(len(cs), "chunks · ยาวสุด", max(len(c["text"]) for c in cs), "ตัวอักษร")
