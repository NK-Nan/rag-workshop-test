"""แสดงคำถามสำหรับ /ask-eval โดยไม่แสดงเฉลย (ไม่มี gold_doc_ids, answer_keywords, must_not)

python -m rag.questions dev 10
"""
import json
import sys

from rag.evaluate import load_questions


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    for q in load_questions(split)[:n]:
        print(json.dumps({"qid": q["qid"], "question": q["question"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
