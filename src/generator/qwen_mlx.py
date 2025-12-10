import os, json, textwrap, glob
from typing import List
from mlx_lm import load, generate

IN_DIR  = "runs/retrieval"
OUT_DIR = "runs/gen"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL = "Qwen/Qwen2.5-3B-Instruct"  # swap to 7B later (quantized)

def build_prompt(query: str, passages: List[str]) -> str:
    passages = [p[:1200] for p in passages]  # truncate defensively
    sources = "\n\n".join(f"[{i+1}] {p}" for i,p in enumerate(passages))
    return textwrap.dedent(f"""\
      You are a factual assistant. Use ONLY the sources to answer.
      Cite with [1], [2], [3].

      Question: {query}

      Sources:
      {sources}

      Answer:
    """)

if __name__ == "__main__":
    model, tok = load(MODEL)

    for p in glob.glob(os.path.join(IN_DIR, "*.json")):
        obj = json.load(open(p, "r", encoding="utf-8"))
        qid, q = obj["qid"], obj["query"]
        docs = [d["text"] for d in obj["top_docs"]]
        prompt = build_prompt(q, docs)

        out = generate(model, tok, prompt, max_tokens=512, temperature=0.2, top_p=0.9).strip()
        json.dump({"qid":qid, "answer":out, "used_docids":[d["docid"] for d in obj["top_docs"]]},
                  open(os.path.join(OUT_DIR, f"{qid}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("Generated:", qid)
