import os, json, textwrap, argparse, requests

def build_prompt(q, docs):
    rows = []
    for i, d in enumerate(docs, 1):
        txt = (d.get("text") or d.get("raw") or "").replace("\n", " ")
        rows.append(f"[d{i}] {textwrap.shorten(txt, width=1200, placeholder=' …')}")
    ctx = "\n".join(rows)
    return (
        "You are a careful assistant. Answer ONLY using the provided documents. "
        "If the documents do not contain the answer, say you don't know.\n\n"
        f"Question: {q}\n\nDocuments:\n{ctx}\n\nAnswer:"
    )

def ask_ollama(model, prompt, num_predict=256, temperature=0.2, host=None):
    url = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": int(num_predict),
            "temperature": float(temperature),
        },
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json().get("response", "").strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)             # e.g. runs/retrieval/dev5.topk3.jsonl
    ap.add_argument("--out", required=True)                   # e.g. runs/answers/dev5.qwen7b.jsonl
    ap.add_argument("--model", default="qwen2.5:7b-instruct-q4_K_M")
    ap.add_argument("--num_predict", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.retrieval, "r", encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)
            qid, q, docs = ex["id"], ex["query"], ex["docs"]
            prompt = build_prompt(q, docs)
            ans = ask_ollama(args.model, prompt, num_predict=args.num_predict, temperature=args.temperature)
            print(f"\n### {qid} — {q}\n{ans}\n", flush=True)
            fout.write(json.dumps({"id": qid, "query": q, "answer": ans}, ensure_ascii=False) + "\n")
            print("Wrote:", qid)

if __name__ == "__main__":
    main()
