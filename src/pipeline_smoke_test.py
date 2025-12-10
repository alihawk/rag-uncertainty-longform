#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end smoke test (HPC/Slurm-friendly):
- Download wiki-18 snapshot (HF dataset) -> create 200-doc shard
- Build BM25 index (Pyserini/Lucene)
- Retrieve & write top-k docs
- [Optional] Generate answers with Qwen2.5-7B-Instruct (CUDA 4-bit if possible)
- Extensive logging to file + stdout with timestamps and environment info

Idempotent: will skip work if artifacts exist.
"""

import os
import sys
import re
import gzip
import json
import time
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ---------------- Logging ----------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "smoke.log"

class _StdoutFilter(logging.Filter):
    def filter(self, record):
        # keep INFO and above on console; everything goes to file
        return record.levelno >= logging.INFO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
for h in logging.getLogger().handlers:
    if isinstance(h, logging.StreamHandler):
        h.addFilter(_StdoutFilter())

log = logging.getLogger("pipeline")

# ---------------- Env guardrails ----------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Xms1g -Xmx8g")

# ---------------- Paths & constants ----------------
HF_DATA_REPO = "PeterJinGo/wiki-18-corpus"
SHARDS_DIR = Path("data/wiki18/shards_small")    # 200 docs
INDEX_DIR  = Path("index/bm25_small")
RUNS_DIR   = Path("runs"); RUNS_DIR.mkdir(parents=True, exist_ok=True)

QUERY_FILE = Path("data/queries/dev5.jsonl")
RETR_OUT   = Path("runs/retrieval/dev5.topk3.jsonl")
ANS_OUT    = Path("runs/answers/dev5.qwen7b.jsonl")

TEST_QUERIES = [
    {"id": "q1", "query": "Who is Bridget Moynahan? Provide a short bio."},
    {"id": "q2", "query": "What is the capital of Luxembourg?"},
    {"id": "q3", "query": "When did the United States enter World War I?"},
]

# ---------------- Helpers ----------------
def banner(msg: str):
    line = "=" * 80
    log.info(line); log.info(f"*** {msg} ***"); log.info(line)

def log_sysinfo():
    banner("SYSTEM / ENV INFO")
    # Slurm env summary
    for k in ["SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST",
              "SLURM_STEP_NODELIST", "SLURM_NTASKS", "SLURM_CPUS_PER_TASK",
              "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "CUDA_VISIBLE_DEVICES"]:
        if k in os.environ:
            log.info(f"{k}={os.environ[k]}")
    # HF cache dirs
    for k in ["HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"]:
        if k in os.environ:
            log.info(f"{k}={os.environ[k]}")
    # Java/py versions
    log.info(f"Python: {sys.version.split()[0]}")
    try:
        out = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT, text=True)
        log.info("Java:\n" + out.strip())
    except Exception as e:
        log.warning(f"Java not found in PATH: {e}")
    # GPU info (if torch present)
    try:
        import torch
        log.info(f"torch.cuda.is_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log.info(f"CUDA device count={torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                log.info(f"GPU[{i}] name={torch.cuda.get_device_name(i)}")
    except Exception as e:
        log.info(f"torch not imported / no CUDA: {e}")

def ensure_queries():
    QUERY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not QUERY_FILE.exists():
        with open(QUERY_FILE, "w", encoding="utf-8") as f:
            for q in TEST_QUERIES:
                f.write(json.dumps(q) + "\n")
        log.info(f"Wrote queries to {QUERY_FILE}")
    else:
        log.info(f"Queries already present at {QUERY_FILE}")

def find_jsonl_gz(snapshot_path: Path) -> Path:
    cand = list(snapshot_path.glob("wiki-18.jsonl.gz"))
    if len(cand) == 1:
        return cand[0]
    cand = list(snapshot_path.rglob("*.jsonl.gz"))
    if not cand:
        raise FileNotFoundError("No *.jsonl.gz found in snapshot.")
    return cand[0]

def download_wiki18() -> Path:
    from huggingface_hub import snapshot_download
    banner("DOWNLOAD DATASET (OR USE CACHE)")
    log.info(f"Repo: {HF_DATA_REPO}")
    p = snapshot_download(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=["*.jsonl.gz"],
        local_files_only=False,
        # resume in place; we want robustness if the job is preempted
        tqdm_class=None,
    )
    snapshot = Path(p)
    gz = find_jsonl_gz(snapshot)
    log.info(f"Snapshot dir: {snapshot}")
    log.info(f"GZ: {gz}")
    return gz
def build_small_shards(gz_path: Path, out_dir: Path, n_docs: int = 200):
    banner("BUILD SMALL SHARD")
    if out_dir.exists():
        count = len(list(out_dir.glob("*.jsonl")))
        if count >= 1:
            log.info(f"Shards already present ({count} files) at {out_dir}; skipping.")
            return
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "wiki18_smoke_000.jsonl"

    # Optional sanity: verify gzip magic
    try:
        with open(gz_path, "rb") as f:
            magic = f.read(2)
        if magic != b"\x1f\x8b":
            log.warning(f"{gz_path} does not start with gzip magic; continuing anyway.")
    except Exception as e:
        log.warning(f"Could not read gzip magic: {e}")

    log.info(f"Writing {n_docs} docs -> {out_file}")
    written = 0
    t0 = time.time()

    # Read binary; decode each line with errors='replace' so bad bytes don’t kill the stream
    import gzip as _gzip
    with _gzip.open(gz_path, "rb") as gzf, open(out_file, "w", encoding="utf-8") as out:
        for bline in gzf:
            try:
                line = bline.decode("utf-8", errors="replace")
            except Exception:
                # Extremely rare; skip if decode utterly fails
                continue

            try:
                rec = json.loads(line)
            except Exception:
                # Bad JSON (after replacement) — skip the line
                continue

            _id = rec.get("id") or rec.get("page_id") or str(written)
            text = rec.get("text") or rec.get("contents") or ""
            if not isinstance(text, str):
                try:
                    text = str(text)
                except Exception:
                    continue
            text = text.strip()
            if not text:
                continue

            out.write(json.dumps({"id": str(_id), "contents": text}, ensure_ascii=False) + "\n")
            written += 1
            if written % 50 == 0:
                log.info(f"  ... {written}/{n_docs}")
            if written >= n_docs:
                break

    log.info(f"Wrote {written} docs in {time.time()-t0:.2f}s")


def build_bm25_index(input_dir: Path, index_dir: Path):
    banner("BUILD BM25 INDEX (Pyserini/Lucene)")
    if index_dir.exists() and any(index_dir.iterdir()):
        log.info(f"Index already exists at {index_dir}; skipping.")
        return
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(input_dir),
        "--index", str(index_dir),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "8",
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    log.info("Indexer cmd: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    log.info(f"Index ready at {index_dir}")

def run_retrieval(queries_path: Path, index_dir: Path, out_path: Path, k: int = 1000, keep: int = 3):
    banner("RETRIEVE TOP-K")
    from pyserini.search.lucene import LuceneSearcher
    out_path.parent.mkdir(parents=True, exist_ok=True)
    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(k1=0.9, b=0.4)
    with open(queries_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)
            qid = ex.get("id") or ex.get("qid")
            q = ex["query"]
            hits = searcher.search(q, k)
            kept = []
            for h in hits[:keep]:
                docid = h.docid
                raw = searcher.doc(docid).raw()
                kept.append({"docid": docid, "raw": raw, "bm25": float(h.score)})
            fout.write(json.dumps({"id": qid, "query": q, "docs": kept}, ensure_ascii=False) + "\n")
            log.info(f"[{qid}] kept={len(kept)} best_bm25={kept[0]['bm25'] if kept else 'NA'}")
    log.info(f"Wrote retrieval to {out_path}")

def parse_docs_for_prompt(docs):
    texts = []
    for i, d in enumerate(docs, 1):
        try:
            obj = json.loads(d["raw"])
            text = obj.get("contents", "")
        except Exception:
            text = d.get("raw", "")
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 1200:
            text = text[:1195] + " ..."
        texts.append(f"[d{i}] {text}")
    return "\n".join(texts)

def build_prompt(query, docs, strict=True):
    policy = ("Answer only using the provided documents. If the documents do not contain the answer, say you don't know."
              if strict else
              "Prefer the provided documents; if insufficient, you may use general knowledge.")
    return f"You are a careful assistant. {policy}\n\nQuestion: {query}\n\nDocuments:\n{parse_docs_for_prompt(docs)}\n\nAnswer:"

def try_load_qwen(model_id: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    log.info(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            import bitsandbytes as bnb  # noqa
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=bnb_cfg,
                torch_dtype=torch.bfloat16,
            )
            log.info("Qwen loaded in 4-bit (bitsandbytes) on CUDA.")
            return tokenizer, model, device
        except Exception as e:
            log.warning(f"4-bit load failed; falling back to bf16. Reason: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype=torch.bfloat16
            )
            log.info("Qwen loaded in bf16 on CUDA.")
            return tokenizer, model, device
    # CPU fallback
    model = AutoModelForCausalLM.from_pretrained(model_id)
    log.info("Qwen loaded on CPU (slow).")
    return tokenizer, model, device

def generate_answers(retr_path: Path, out_path: Path, model_id: str, max_new_tokens: int = 256, strict=True):
    banner("GENERATE ANSWERS (Qwen2.5-7B-Instruct)")
    from transformers import TextStreamer
    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer, model, device = try_load_qwen(model_id)

    with open(retr_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)
            qid, q, docs = ex["id"], ex["query"], ex["docs"]
            prompt = build_prompt(q, docs, strict=strict)
            inputs = tokenizer(prompt, return_tensors="pt")
            if device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            # Stream to stdout so you see it live in the Slurm .out
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen = model.generate(
                **inputs,
                do_sample=False,                  # deterministic for smoke
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )
            full_text = tokenizer.decode(gen[0], skip_special_tokens=True)
            answer = full_text.split("Answer:", 1)[-1].strip() if "Answer:" in full_text else full_text.strip()
            fout.write(json.dumps({"id": qid, "query": q, "answer": answer}, ensure_ascii=False) + "\n")
            log.info(f"[{qid}] wrote answer (len={len(answer)})")

    log.info(f"Answers written -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_docs", type=int, default=200)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--skip_gen", action="store_true")
    ap.add_argument("--non_strict", action="store_true", help="Allow general knowledge beyond docs (for quick sanity)")
    args = ap.parse_args()

    banner("SMOKE TEST START")
    log_sysinfo()
    log.info(f"Args: {vars(args)}")

    try:
        ensure_queries()
        gz = download_wiki18()
        build_small_shards(gz, SHARDS_DIR, n_docs=args.n_docs)
        build_bm25_index(SHARDS_DIR, INDEX_DIR)
        RETR_OUT.parent.mkdir(parents=True, exist_ok=True)
        run_retrieval(QUERY_FILE, INDEX_DIR, RETR_OUT, k=args.k, keep=args.keep)
        if not args.skip_gen:
            ANS_OUT.parent.mkdir(parents=True, exist_ok=True)
            generate_answers(RETR_OUT, ANS_OUT, args.model, strict=not args.non_strict)
    except subprocess.CalledProcessError as e:
        log.exception("Subprocess failed.")
        sys.exit(e.returncode)
    except Exception:
        log.exception("Pipeline failed.")
        sys.exit(1)
    finally:
        banner("SMOKE TEST DONE")
        log.info(f"Log file: {LOG_FILE.absolute()}")

if __name__ == "__main__":
    main()
