# src/retriever/retrieve_rerank.py
import os, json, argparse
from typing import List, Tuple
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import CrossEncoder

def bm25_topk(index_dir: str, query: str, k: int = 1000) -> List[Tuple[str, str, float]]:
    s = LuceneSearcher(index_dir)
    s.set_bm25(k1=0.9, b=0.4)
    hits = s.search(query, k)
    out = []
    for h in hits:
        raw = s.doc(h.docid).raw()  # JSON string: {"id":"...", "contents":"..."}
        try:
            text = json.loads(raw).get("contents", raw)
        except Exception:
            text = raw
        out.append((h.docid, text, float(h.score)))  # ensure plain float
    return out

def rerank_ce(ce: CrossEncoder, query: str, docs: List[Tuple[str,str,float]], keep: int = 3):
    if not docs:
        return []
    pairs = [(query, d[1]) for d in docs]
    scores = ce.predict(pairs)  # numpy array (float32)
    ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)[:keep]
    return [
        {"docid": d[0][0], "text": d[0][1], "bm25": float(d[0][2]), "ce": float(s)}
        for d, s in ranked
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="JSONL with {id, query}")
    ap.add_argument("--index",   required=True, help="Pyserini Lucene index dir")
    ap.add_argument("--out",     required=True, help="Output JSONL (one line per query)")
    ap.add_argument("--k",       type=int, default=1000)
    ap.add_argument("--keep",    type=int, default=3)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")  # load once

    with open(args.queries, "r", encoding="utf-8") as f_in, \
         open(args.out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            ex = json.loads(line)
            qid, query = ex["id"], ex["query"]
            bm25 = bm25_topk(args.index, query, args.k)
            top = rerank_ce(ce, query, bm25, args.keep)
            f_out.write(json.dumps({"id": qid, "query": query, "docs": top}, ensure_ascii=False) + "\n")
            print("Saved:", qid)

if __name__ == "__main__":
    main()
