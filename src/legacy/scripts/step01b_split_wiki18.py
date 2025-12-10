# src/scripts/step01b_split_wiki18.py
import os, gzip, json, argparse, glob, io

def find_gz(src_arg: str | None):
    if src_arg and os.path.isfile(src_arg):
        return src_arg
    SNAP = "/Users/ali/.cache/huggingface/hub/datasets--PeterJinGo--wiki-18-corpus/snapshots/69c1c00ffe7c5554c68d8548355cb22e46aabc51"
    cands = glob.glob(os.path.join(SNAP, "**", "*.jsonl.gz"), recursive=True)
    if not cands:
        raise FileNotFoundError("No *.jsonl.gz under snapshot. Pass --src <path/to/wiki-18.jsonl.gz>.")
    return max(cands, key=os.path.getsize)

def open_text_gzip(path: str):
    # Open gz as binary, then decode with a tolerant wrapper
    gz = gzip.open(path, "rb")
    return io.TextIOWrapper(gz, encoding="utf-8", errors="replace", newline="")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="Path to wiki-18.jsonl.gz (optional)")
    ap.add_argument("--out_dir", default="data/wiki18/shards")
    ap.add_argument("--lines_per_shard", type=int, default=50_000)
    args = ap.parse_args()

    gz_path = find_gz(args.src)
    os.makedirs(args.out_dir, exist_ok=True)
    print("Reading:", gz_path)
    print("Writing shards to:", args.out_dir)

    shard_idx, line_idx, kept = 0, 0, 0
    out_path = os.path.join(args.out_dir, f"wiki18_shard_{shard_idx:05d}.jsonl")
    out_f = open(out_path, "w", encoding="utf-8")

    with open_text_gzip(gz_path) as gzf:
        for raw in gzf:
            if not raw:
                continue
            # tolerate stray NULs/invalid chars already handled by errors="replace"
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                # Skip malformed line
                continue

            text = obj.get("text") or obj.get("contents") or ""
            if not text:
                continue
            doc_id = str(obj.get("id", line_idx))

            rec = {"id": doc_id, "contents": text}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            line_idx += 1

            if kept % 10_000 == 0:
                print(f"... wrote {kept} docs (current shard {shard_idx})")

            if kept % args.lines_per_shard == 0:
                out_f.close()
                shard_idx += 1
                out_f = open(os.path.join(args.out_dir, f"wiki18_shard_{shard_idx:05d}.jsonl"),
                             "w", encoding="utf-8")

    out_f.close()
    print(f"Done. Wrote {shard_idx + 1} shard(s). Total docs kept: {kept}")

if __name__ == "__main__":
    main()
