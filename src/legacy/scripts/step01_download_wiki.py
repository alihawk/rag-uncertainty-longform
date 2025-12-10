# src/scripts/step01_download_wiki.py
import os, json, itertools
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = "data/wiki18"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Stream (no full 5GB download)
ds_stream = load_dataset("PeterJinGo/wiki-18-corpus", split="train", streaming=True)

N = 200  # try 500–1000 to start
print("SCRIPT:", __file__)
print("N =", N)

out_path = os.path.join(OUT_DIR, f"wiki18_head_{N}.jsonl")

it = iter(ds_stream)  # create ONE iterator

with open(out_path, "w", encoding="utf-8") as f:
    for row in tqdm(itertools.islice(it, N), total=N, desc="Writing JSONL"):
        # dataset fields are typically: id, url, title, text
        text = row.get("text") or row.get("contents") or ""
        if not text:
            continue
        doc_id = str(row.get("id")) if row.get("id") is not None else str(abs(hash(text)) % 10**12)
        f.write(json.dumps({"id": doc_id, "contents": text}, ensure_ascii=False) + "\n")

print("Wrote:", out_path)

