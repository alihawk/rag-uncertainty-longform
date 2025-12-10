import os, subprocess, sys, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder with *.jsonl shards")
    ap.add_argument("--index", required=True, help="Output Lucene index folder")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)
    args = ap.parse_args()

    os.makedirs(args.index, exist_ok=True)
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", args.input,
        "--index", args.index,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(args.threads),
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Index built at", args.index)

if __name__ == "__main__":
    main()
