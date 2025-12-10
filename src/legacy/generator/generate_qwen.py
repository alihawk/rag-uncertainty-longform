# at the top
import os, json, argparse
from textwrap import shorten
from mlx_lm import load, generate   # same imports

def build_prompt(query, docs):
    lines = []
    for i, d in enumerate(docs, 1):
        text = (d.get("text") or "").replace("\n", " ")
        lines.append(f"[d{i}] {shorten(text, width=1200, placeholder=' ...')}")
    context = "\n".join(lines)
    return (
        "You are a careful assistant. Answer only using the provided documents. "
        "If the documents do not contain the answer, say you don't know.\n\n"
        f"Question: {query}\n\nDocuments:\n{context}\n\nAnswer:"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=384)
    # keep --quantize in CLI to avoid breaking your command, but ignore it
    ap.add_argument("--quantize", default=None)
    args = ap.parse_args()

    # ✅ Correct order: (model, tokenizer)
    model, tokenizer = load(args.model)  # no quantize kw for this mlx_lm version

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fout, \
         open(args.retrieval, "r", encoding="utf-8") as fin:
        for line in fin:
            ex = json.loads(line)
            qid, query, docs = ex["id"], ex["query"], ex["docs"]
            prompt = build_prompt(query, docs)

            # Optional: explicit end tokens fallback
            end_tokens = {tokenizer.eos_token_id} if getattr(tokenizer, "eos_token_id", None) is not None else None

            out_text = ""
            for chunk in generate(model, tokenizer, prompt, max_tokens=int(args.max_tokens)):
                out_text += chunk

            fout.write(json.dumps({"id": qid, "query": query, "answer": out_text}, ensure_ascii=False) + "\n")
            print("Wrote:", qid)

if __name__ == "__main__":
    main()
