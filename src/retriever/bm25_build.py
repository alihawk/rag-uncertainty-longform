import os, subprocess, sys

INPUT_DIR = "data/wiki18"
INDEX_DIR = "index/bm25"
os.makedirs(INDEX_DIR, exist_ok=True)

cmd = [
    sys.executable, "-m", "pyserini.index.lucene",
    "--collection", "JsonCollection",
    "--input", INPUT_DIR,
    "--index", INDEX_DIR,
    "--generator", "DefaultLuceneDocumentGenerator",
    "--threads", "8",
    "--storePositions", "--storeDocvectors", "--storeRaw"
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print("Index built at", INDEX_DIR)
