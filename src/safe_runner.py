# safe_runner.py

import json, torch, csv
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "safe-ai/SAFE-Classifier"

def run_safe_scores(answers_path: Path, out_jsonl: Path, out_csv: Path):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    rows = []

    with open(answers_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            qid = ex["id"]
            q = ex["query"]
            ans = ex["answer"]

            text = f"question: {q}\nanswer: {ans}"
            inp = tok(text, return_tensors="pt", truncation=True, max_length=1024)
            if torch.cuda.is_available():
                inp = {k: v.cuda() for k, v in inp.items()}

            with torch.no_grad():
                logits = mdl(**inp).logits
                score = float(torch.softmax(logits, dim=-1)[0][1])

            rows.append({
                "id": qid,
                "query": q,
                "answer": ans,
                "safe_score": score
            })

    # Write JSONL
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","safe_score"])
        for r in rows: w.writerow([r["id"], r["safe_score"]])
