import os, json, glob
os.makedirs("runs/safe", exist_ok=True)

# adjust import per SAFE repo
from safe.scorer import SafeScorer
scorer = SafeScorer()

for p in glob.glob("runs/gen/*.json"):
    gen = json.load(open(p, "r", encoding="utf-8"))
    qid = gen["qid"]
    ret = json.load(open(f"runs/retrieval/{qid}.json", "r", encoding="utf-8"))

    res = scorer.score(
        question=ret["query"],
        answer=gen["answer"],
        contexts=[d["text"] for d in ret["top_docs"]],
    )
    json.dump({"qid": qid, "safe": res},
              open(f"runs/safe/{qid}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("SAFE:", qid)
