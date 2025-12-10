import os, json
from typing import List, Tuple
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import CrossEncoder

INDEX = "index/bm25"
OUT_DIR = "runs/retrieval"
os.makedirs(OUT_DIR, exist_ok=True)

def bm25_topk(query: str, k: int = 1000) -> List[Tuple[str, str, float]]:
    s = LuceneSearcher(INDEX)
    s.set_bm25(k1=0.9, b=0.4)
    hits = s.search(query, k)
    return [(h.docid, s.doc(h.docid).raw(), h.score) for h in hits]

def rerank_ce(query: str, docs, keep=3):
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    pairs = [(query, d[1]) for d in docs]
    scores = ce.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:keep]
    return [{"docid":d[0][0], "text":d[0][1], "bm25":d[0][2], "ce":s} for d,s in ranked]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="jsonl with qid,query")
    args = ap.parse_args()

    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            qid, q = ex["qid"], ex["query"]
            top = rerank_ce(q, bm25_topk(q, 1000), keep=3)
            with open(os.path.join(OUT_DIR, f"{qid}.json"), "w", encoding="utf-8") as out:
                json.dump({"qid":qid, "query":q, "top_docs":top}, out, ensure_ascii=False, indent=2)
            print("Saved:", qid)
