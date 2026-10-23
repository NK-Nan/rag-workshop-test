"""ค้นคืน chunk ที่เกี่ยวข้องกับคำถาม (เวิร์กช็อป 2 และ 4) ทำงานในเครื่องทั้งหมด ไม่เรียก API

python -m rag.retrieve "คำถาม" -k 4 [--method tfidf|hybrid] [--include-superseded] [--json]
"""
import argparse
import json

import numpy as np
from pythainlp.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from rag.chunk import chunk_docs

METHODS = ("tfidf", "hybrid")
TFIDF_WEIGHT = 0.7  # hybrid = TFIDF_WEIGHT * tfidf + (1 - TFIDF_WEIGHT) * bm25 หลัง normalize แต่ละฝั่งด้วยค่าสูงสุด

# คำพ้องสำหรับขยายคำค้น (hybrid เท่านั้น) ได้จากคำถามชุด dev ที่ผลอันดับ 1 ผิด (tfidf: Q35 · hybrid ก่อนเพิ่มคำพ้อง: Q30)
SYNONYMS = {
    "อีวี": "รถยนต์ไฟฟ้า",  # Q35
    "เสียบชาร์จ": "จุดชาร์จ",  # Q35
    "โน้ตบุ๊ก": "คอมพิวเตอร์",  # Q30
    "หาย": "สูญหาย",  # Q30
}


def index_text(c: dict) -> str:
    return f"{c['title']}\n{c['section']}\n{c['text']}"


def tokenize(text: str) -> list[str]:
    return [t for t in word_tokenize(text.lower(), engine="newmm") if t.strip() and any(ch.isalnum() for ch in t)]


def expand_query(query: str) -> str:
    extra = [v for k, v in SYNONYMS.items() if k in query]
    return " ".join([query, *extra])


def normalize(scores: np.ndarray) -> np.ndarray:
    top = scores.max()
    return scores / top if top > 0 else scores


class Retriever:
    def __init__(self, chunks: list[dict] | None = None, method: str = "tfidf"):
        """method="tfidf": TF-IDF ระดับตัวอักษร (W2) · method="hybrid": TF-IDF + BM25 บนคำที่ตัดด้วย newmm และขยายคำค้น (W4)"""
        if method not in METHODS:
            raise ValueError(f"method ต้องเป็นหนึ่งใน {METHODS}")
        self.method = method
        self.chunks = chunk_docs() if chunks is None else chunks
        texts = [index_text(c) for c in self.chunks]
        # ภาษาไทยไม่เว้นวรรคระหว่างคำ จึงใช้ n-gram ระดับตัวอักษรแทนการตัดคำ
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(texts)
        if method == "hybrid":
            self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def scores(self, query: str) -> np.ndarray:
        if self.method == "tfidf":
            # TfidfVectorizer ทำ L2 normalize แล้ว dot product จึงเท่ากับ cosine similarity
            return linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        query = expand_query(query)
        tfidf = linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        bm25 = np.asarray(self.bm25.get_scores(tokenize(query)))
        return TFIDF_WEIGHT * normalize(tfidf) + (1 - TFIDF_WEIGHT) * normalize(bm25)

    def search(self, query: str, k: int = 4, include_superseded: bool = True) -> list[dict]:
        """คืน chunk k อันที่คะแนนสูงสุด เรียงจากมากไปน้อย แต่ละอันเพิ่มช่อง "score" """
        scores = self.scores(query)
        results = []
        for i in scores.argsort()[::-1]:
            c = self.chunks[i]
            if not include_superseded and c["status"] != "active":
                continue
            results.append({**c, "score": float(scores[i])})
            if len(results) == k:
                break
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--method", default="tfidf", choices=METHODS)
    ap.add_argument("--include-superseded", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = Retriever(method=a.method).search(a.query, a.k, a.include_superseded)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    for r in res:
        print(f"[{r['score']:.3f}] {r['chunk_id']} ({r['status']}) {r['title']} · {r['section']}")
        print("   " + r["text"].split("\n", 1)[-1].replace("\n", "\n   "))


if __name__ == "__main__":
    main()
