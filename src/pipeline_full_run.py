#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full wiki-18 pipeline (HPC/Slurm-friendly)
Stages:
  1) Data: download → shard → index (BM25)
  2) Queries: sample N from JSONL (idempotent)
  3) Retrieval: BM25 top-Kfirst (e.g., 1000)
  4) Rerank: cross-encoder (ms-marco-MiniLM-L6-v2), keep top-Kkeep (e.g., 3)
  5) Generation: Qwen2.5-7B-Instruct (bf16 if CUDA) with strict doc-only policy
  6) UE: White-box=MARS, Black-box=Eccentricity (separate logs + JSONL)
  7) Report: Markdown table

All stages are resumable and write detailed logs.
"""

import os, sys, re, gzip, json, time, random, logging, argparse, subprocess, threading
from pathlib import Path
from datetime import datetime

# ---------------- Logging ----------------
LOG_DIR = Path("logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "full_run.log"

class _StdoutFilter(logging.Filter):
    def filter(self, record): return record.levelno >= logging.INFO

def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
        force=True,
    )
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler):
            h.addFilter(_StdoutFilter())
            try: h.flush = lambda: sys.stdout.flush()
            except Exception: pass
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception: pass

_setup_logging()
log = logging.getLogger("pipeline")

# heartbeat thread
_stop_hb = threading.Event()
def _heartbeat(period=30):
    while not _stop_hb.is_set():
        log.info("[HEARTBEAT] job is alive..."); sys.stdout.flush()
        _stop_hb.wait(period)
hb = threading.Thread(target=_heartbeat, daemon=True); hb.start()

def _stage_mark(name: str, status: str):
    Path(f"logs/stage_{name}.{status}").write_text(datetime.now().isoformat(timespec="seconds"))

def _banner(msg: str):
    line = "="*80
    log.info(line); log.info(f"*** {msg} ***"); log.info(line); sys.stdout.flush()

# ---------------- Env guardrails ----------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Xms1g -Xmx8g")

# ---------------- Paths ----------------
HF_DATA_REPO = "PeterJinGo/wiki-18-corpus"
FULL_SHARDS_DIR = Path("data/wiki18/shards_full")
FULL_INDEX_DIR  = Path("index/bm25_full")
RUNS_DIR = Path("runs"); RUNS_DIR.mkdir(parents=True, exist_ok=True)
RETR_DIR = RUNS_DIR / "retrieval"; RETR_DIR.mkdir(parents=True, exist_ok=True)
ANS_DIR  = RUNS_DIR / "answers";   ANS_DIR.mkdir(parents=True, exist_ok=True)
UE_DIR   = RUNS_DIR / "ue";        UE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("reports"); REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# UE log files
MARS_LOG = LOG_DIR / "mars.log"
ECC_LOG  = LOG_DIR / "eccentricity.log"

# ---------------- Sysinfo ----------------
def log_sysinfo():
    _banner("SYSTEM / ENV INFO")
    for k in ["SLURM_JOB_ID","SLURM_JOB_NAME","SLURM_JOB_NODELIST","SLURM_STEP_NODELIST",
              "SLURM_NTASKS","SLURM_CPUS_PER_TASK","SLURM_MEM_PER_NODE","SLURM_MEM_PER_CPU",
              "CUDA_VISIBLE_DEVICES"]:
        if k in os.environ: log.info(f"{k}={os.environ[k]}")
    for k in ["HF_HOME","HUGGINGFACE_HUB_CACHE","TRANSFORMERS_CACHE","JAVA_HOME"]:
        if k in os.environ: log.info(f"{k}={os.environ[k]}")
    log.info(f"Python: {sys.version.split()[0]}")
    try:
        out = subprocess.check_output(["java","-version"], stderr=subprocess.STDOUT, text=True)
        log.info("Java:\n" + out.strip())
    except Exception as e:
        log.warning(f"Java not found: {e}")
    try:
        import torch
        log.info(f"torch.cuda.is_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log.info(f"CUDA device count={torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                log.info(f"GPU[{i}] name={torch.cuda.get_device_name(i)}")
    except Exception as e:
        log.info(f"torch info unavailable: {e}")
    sys.stdout.flush()

# ---------------- Data helpers ----------------
def _find_jsonl_gz(snapshot_path: Path) -> Path:
    cand = list(snapshot_path.glob("wiki-18.jsonl.gz"))
    if len(cand) == 1: return cand[0]
    cand = list(snapshot_path.rglob("*.jsonl.gz"))
    if not cand: raise FileNotFoundError("No *.jsonl.gz found in snapshot.")
    return cand[0]

def download_wiki18() -> Path:
    from huggingface_hub import snapshot_download
    _banner("DOWNLOAD DATASET (OR USE CACHE)")
    _stage_mark("download","start")
    log.info(f"Repo: {HF_DATA_REPO}")
    p = snapshot_download(repo_id=HF_DATA_REPO, repo_type="dataset",
                          allow_patterns=["*.jsonl.gz"], local_files_only=False, tqdm_class=None)
    snapshot = Path(p)
    gz = _find_jsonl_gz(snapshot)
    log.info(f"Snapshot dir: {snapshot}")
    log.info(f"GZ: {gz}")
    _stage_mark("download","done")
    return gz

def stream_full_to_shards(gz_path: Path, out_dir: Path, shard_size: int = 100_000):
    _banner("BUILD FULL SHARDS"); _stage_mark("shard","start")
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.jsonl"))
    if existing:
        log.info(f"Shards already present ({len(existing)}) at {out_dir}; skipping.")
        _stage_mark("shard","done"); return
    total, shard_idx, written_this = 0, 0, 0
    current = open(out_dir / f"wiki18_full_{shard_idx:03d}.jsonl", "w", encoding="utf-8")
    t0 = time.time()
    import json as _json, gzip as _gzip
    with _gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as gzf:
        for line in gzf:
            try: rec = _json.loads(line)
            except Exception: continue
            _id = rec.get("id") or rec.get("page_id") or str(total)
            text = rec.get("text") or rec.get("contents") or ""
            if not isinstance(text, str):
                try: text = str(text)
                except Exception: continue
            text = text.strip()
            if not text: continue
            current.write(_json.dumps({"id": str(_id), "contents": text}, ensure_ascii=False) + "\n")
            total += 1; written_this += 1
            if total % 50_000 == 0: log.info(f"  ... streamed {total} docs")
            if written_this >= shard_size:
                current.close(); shard_idx += 1; written_this = 0
                current = open(out_dir / f"wiki18_full_{shard_idx:03d}.jsonl", "w", encoding="utf-8")
    current.close()
    log.info(f"Full stream complete: {total} docs in {time.time()-t0:.1f}s, {shard_idx+1} shard(s).")
    _stage_mark("shard","done")

def build_bm25_index(input_dir: Path, index_dir: Path, threads: int = 8):
    _banner("BUILD BM25 INDEX (Pyserini/Lucene) [FULL]"); _stage_mark("index","start")
    if index_dir.exists() and any(index_dir.iterdir()):
        log.info(f"Index already exists at {index_dir}; skipping.")
        _stage_mark("index","done"); return
    cmd = [sys.executable, "-m", "pyserini.index.lucene",
           "--collection","JsonCollection", "--input", str(input_dir),
           "--index", str(index_dir), "--generator","DefaultLuceneDocumentGenerator",
           "--threads", str(threads), "--storePositions","--storeDocvectors","--storeRaw"]
    log.info("Indexer cmd: " + " ".join(cmd)); sys.stdout.flush()
    subprocess.run(cmd, check=True)
    log.info(f"Index ready at {index_dir}")
    _stage_mark("index","done")

# -------- Query loading and sampling --------
def _extract_query_text(obj):
    for k in ("query","question","prompt","instruction","text","claim"):
        if k in obj and isinstance(obj[k], str) and obj[k].strip(): return obj[k].strip()
    for v in obj.values():
        if isinstance(v, str) and v.strip(): return v.strip()
    return None

def sample_queries_from_file(src: Path, out_path: Path, n: int = 50, seed: int = 42):
    _banner("LOAD + SAMPLE QUERIES"); _stage_mark("sample","start")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        log.info(f"Sampled queries already present at {out_path}; skipping.")
        _stage_mark("sample","done"); return
    items = []
    with open(src, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try: obj = json.loads(line)
            except Exception: continue
            q = _extract_query_text(obj)
            if not q: continue
            qid = str(obj.get("id") or obj.get("_id") or obj.get("qid") or f"s{ i:08d }")
            items.append((qid, q))
    if len(items) < n: raise RuntimeError(f"Source has only {len(items)} usable queries, need {n}")
    random.Random(seed).shuffle(items)
    picked = items[:n]
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, (qid, q) in enumerate(picked, 1):
            sid = f"b{idx:03d}"; f.write(json.dumps({"id": sid, "orig_id": qid, "query": q}, ensure_ascii=False) + "\n")
    log.info(f"Wrote {n} sampled queries -> {out_path}")
    _stage_mark("sample","done")

# -------- First-stage retrieval + RERANKER (NEW) --------
def _doc_text_from_raw(raw: str) -> str:
    try:
        obj = json.loads(raw); return str(obj.get("contents","")).strip()
    except Exception:
        return str(raw).strip()

def run_retrieval_with_rerank(
    queries_path: Path, index_dir: Path, out_path: Path,
    k_first: int = 1000, k_keep: int = 3, ce_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size: int = 64
):
    """BM25 top-k_first → CE rerank → keep k_keep docs. Resumable."""
    _banner("RETRIEVE + RERANK [BM25 → CrossEncoder]"); _stage_mark("retrieve","start")
    from pyserini.search.lucene import LuceneSearcher
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    existing = {}
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try: ex = json.loads(line); existing[ex["id"]] = ex
                except Exception: pass
        log.info(f"Resume: found {len(existing)} rows")

    # CE model
    ce = None
    try:
        from sentence_transformers import CrossEncoder
        import torch as _torch
        ce = CrossEncoder(ce_model, device="cuda")

        log.info(f"CrossEncoder ready: {ce_model}")
    except Exception as e:
        log.warning(f"[RERANK] CrossEncoder unavailable ({e}); will keep BM25 top-{k_keep} as-is.")

    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(k1=0.9, b=0.4)

    new = 0; skipped = 0
    with open(queries_path, "r", encoding="utf-8") as fin, \
         open(out_path, "a", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)
            qid, q = ex["id"], ex["query"]
            if qid in existing: skipped += 1; continue

            hits = searcher.search(q, k_first)
            if not hits:
                fout.write(json.dumps({"id": qid, "query": q, "docs": []}, ensure_ascii=False) + "\n")
                new += 1; log.info(f"[{qid}] no hits"); continue

            # Build candidate pairs for CE
            cands = []
            for h in hits:
                raw = searcher.doc(h.docid).raw()
                text = _doc_text_from_raw(raw)
                cands.append({"docid": h.docid, "bm25": float(h.score), "raw": raw, "text": text})

            # Rerank or fallback
            if ce is not None:
                pairs = [(q, c["text"]) for c in cands]
                # batch scoring
                try:
                    scores = ce.predict(pairs, batch_size=batch_size, convert_to_tensor=False)
                except Exception as e:
                    log.warning(f"[{qid}] CE.predict failed: {e}; fallback to BM25")
                    scores = None
                if scores is not None:
                    for c, s in zip(cands, scores): c["ce"] = float(s)
                    cands.sort(key=lambda x: x.get("ce", -1e9), reverse=True)
                else:
                    cands.sort(key=lambda x: x["bm25"], reverse=True)
            else:
                cands.sort(key=lambda x: x["bm25"], reverse=True)

            kept = [{"docid": c["docid"], "raw": c["raw"], "bm25": c["bm25"], "ce": c.get("ce", None)}
                    for c in cands[:k_keep]]

            fout.write(json.dumps({"id": qid, "query": q, "docs": kept}, ensure_ascii=False) + "\n")
            new += 1
            best = kept[0] if kept else {}
            log.info(f"[{qid}] kept={len(kept)} best_bm25={best.get('bm25','NA')} best_ce={best.get('ce','NA')}")

    log.info(f"Wrote retrieval+rerank -> {out_path} (new={new}, skipped={skipped})")
    _stage_mark("retrieve","done")

# -------- Prompting / Generation (unchanged core) --------
def _parse_docs_for_prompt(docs):
    texts = []
    for i, d in enumerate(docs, 1):
        try:
            obj = json.loads(d["raw"]); text = obj.get("contents","")
        except Exception:
            text = d.get("raw","")
        text = re.sub(r"\s+"," ", text).strip()
        if len(text) > 1200: text = text[:1195] + " ..."
        texts.append(f"[d{i}] {text}")
    return "\n".join(texts)

def _build_prompt(query, docs, strict=True):
    policy = ("Answer only using the provided documents. If the documents do not contain the answer, say you don't know."
              if strict else
              "Prefer the provided documents; if insufficient, you may use general knowledge.")
    return f"You are a careful assistant. {policy}\n\nQuestion: {query}\n\nDocuments:\n{_parse_docs_for_prompt(docs)}\n\nAnswer:"

def try_load_qwen_bf16(model_id: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _banner(f"LOAD MODEL bf16 (no quant): {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if torch.cuda.is_available():
        mdl = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
        log.info("Qwen loaded in bf16 on CUDA."); return tok, mdl, "cuda"
    mdl = AutoModelForCausalLM.from_pretrained(model_id)
    log.info("Qwen loaded on CPU (fp32)."); return tok, mdl, "cpu"

def generate_answers_with_resume(retr_path: Path, out_path: Path, model_id: str,
                                 max_new_tokens: int = 256, strict=True):
    _banner("GENERATE ANSWERS (Qwen2.5-7B-Instruct) [FULL, RESUMABLE]")
    _stage_mark("generate","start")
    from transformers import TextStreamer
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try: done.add(json.loads(line)["id"])
                except Exception: pass
        log.info(f"Resume: {len(done)} answers already present")

    tokenizer, model, device = try_load_qwen_bf16(model_id)

    new = 0
    with open(retr_path, "r", encoding="utf-8") as fin, \
         open(out_path, "a", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)
            qid, q, docs = ex["id"], ex["query"], ex["docs"]
            if qid in done: continue

            prompt = _build_prompt(q, docs, strict=strict)
            inputs = tokenizer(prompt, return_tensors="pt")
            if device == "cuda": inputs = {k: v.cuda() for k, v in inputs.items()}

            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen = model.generate(
                **inputs, do_sample=False, max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )
            full_text = tokenizer.decode(gen[0], skip_special_tokens=True)
            answer = full_text.split("Answer:", 1)[-1].strip() if "Answer:" in full_text else full_text.strip()

            fout.write(json.dumps({
                "id": qid, "query": q, "answer": answer, "docs": docs,
                "meta": {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "model": model_id,
                    "dtype": "bf16" if device == "cuda" else "fp32-cpu",
                    "max_new_tokens": max_new_tokens
                }
            }, ensure_ascii=False) + "\n")
            new += 1; log.info(f"[{qid}] wrote answer (len={len(answer)})")

    log.info(f"Answers written -> {out_path} (new={new}, skipped={len(done)})")
    _stage_mark("generate","done")

# -------- UE: MARS (white-box) + Eccentricity (black-box) --------
def run_mars_whitebox(answers_path: Path, out_path: Path, model_id: str):
    """
    Compute MARS per QA pair.
    1) Try TruthTorchLM if installed.
    2) Else fallback: compute NLL(answer | prompt) correctly.
    """

    _banner("UE WHITE-BOX: MARS"); _stage_mark("mars","start")

    with open(MARS_LOG, "a", encoding="utf-8") as flog:
        flog.write(f"[{datetime.now().isoformat(timespec='seconds')}] MARS start model={model_id}\n")

    # Try TruthTorchLM
    used_tt = False
    try:
        from truthtorchlm import mars as tt_mars
        used_tt = True
    except Exception as e:
        with open(MARS_LOG, "a", encoding="utf-8") as flog:
            flog.write(f"TruthTorchLM not available: {e}\n")

    # Prepare fallback LM
    tok, mdl = None, None
    if not used_tt:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch   # <-- REQUIRED FIX

            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
        except Exception as e:
            with open(MARS_LOG, "a", encoding="utf-8") as flog:
                flog.write(f"Fallback LM failed: {e}\n")

    results = []

    with open(answers_path, "r", encoding="utf-8") as fin:
        for line in fin:
            ex = json.loads(line)
            qid = ex["id"]
            q = ex["query"]
            ans = ex.get("answer","")
            docs = ex.get("docs", [])

            score, details = None, {}

            # 1) Try TruthTorchLM
            if used_tt:
                try:
                    score, details = tt_mars.compute(
                        question=q,
                        answer=ans,
                        docs=[d.get("raw","") for d in docs],
                        model_name=model_id
                    )
                except Exception as e:
                    with open(MARS_LOG, "a", encoding="utf-8") as flog:
                        flog.write(f"[{qid}] TruthTorchLM error: {e}\n")

            # 2) FALLBACK: correct NLL
            if score is None and tok is not None and mdl is not None:
                try:
                    import torch

                    # Build prompt
                    prompt = _build_prompt(q, docs, strict=True)
                    full = prompt + "\n\nAnswer: " + ans

                    # Tokenize whole sequence
                    enc = tok(full, return_tensors="pt")
                    if torch.cuda.is_available():
                        enc = {k: v.cuda() for k, v in enc.items()}

                    # Identify answer token positions
                    prompt_ids = tok(prompt + "\n\nAnswer:", return_tensors="pt")["input_ids"][0]
                    full_ids = enc["input_ids"][0]
                    ans_len = full_ids.size(0) - prompt_ids.size(0)

                    labels = full_ids.clone()
                    labels[: prompt_ids.size(0)] = -100        # mask prompt tokens

                    out = mdl(**enc, labels=labels.unsqueeze(0))
                    nll = float(out.loss.detach().cpu())

                    score = -nll
                    details = {"fallback": "correct_nll"}

                except Exception as e:
                    with open(MARS_LOG, "a", encoding="utf-8") as flog:
                        flog.write(f"[{qid}] fallback MARS failed: {e}\n")

            results.append({"id": qid, "mars_score": score, "mars_details": details})

    # Save output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r) + "\n")

    with open(MARS_LOG, "a", encoding="utf-8") as flog:
        flog.write(f"[{datetime.now().isoformat(timespec='seconds')}] MARS done → {out_path}\n")

    _stage_mark("mars","done")


def run_eccentricity_blackbox(answers_path: Path, out_path: Path, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Black-box: embed answer vs doc context; eccentricity = distance to context centroid (z-scored)."""
    _banner("UE BLACK-BOX: Eccentricity"); _stage_mark("ecc","start")
    with open(ECC_LOG, "a", encoding="utf-8") as flog:
        flog.write(f"[{datetime.now().isoformat(timespec='seconds')}] ECC start embed={embed_model}\n")

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import math
    except Exception as e:
        with open(ECC_LOG, "a", encoding="utf-8") as flog:
            flog.write(f"sentence-transformers not available: {e}\n")
        # still write empty file
        Path(out_path).write_text("")
        _stage_mark("ecc","done"); return

    try:
        model = SentenceTransformer(embed_model, device="cuda")

    except Exception as e:
        with open(ECC_LOG, "a", encoding="utf-8") as flog:
            flog.write(f"Embedding model load failed: {e}\n")
        Path(out_path).write_text(""); _stage_mark("ecc","done"); return

    rows = []
    with open(answers_path, "r", encoding="utf-8") as fin:
        for line in fin:
            ex = json.loads(line)
            qid, q, ans, docs = ex["id"], ex["query"], ex.get("answer",""), ex.get("docs",[])
            # Build a short “context” summary from top-3 docs
            ctx_texts = []
            for d in docs:
                t = _doc_text_from_raw(d.get("raw",""))
                if t: ctx_texts.append(t[:1000])
            if not ctx_texts:
                rows.append({"id": qid, "ecc": None, "ecc_z": None, "n_ctx": 0})
                continue
            # Embeddings
            vec_ans = model.encode([ans], normalize_embeddings=True)[0]
            vec_ctx = model.encode(ctx_texts, normalize_embeddings=True)
            centroid = vec_ctx.mean(axis=0)
            # cosine distance since normalized
            cos_sim = float((vec_ans * centroid).sum())
            cos_dist = 1.0 - cos_sim
            rows.append({"id": qid, "ecc": cos_dist, "ecc_z": None, "n_ctx": len(ctx_texts)})

    # Batch z-score over this run (you can later calibrate on dev if you prefer)
    try:
        import numpy as _np
        vals = _np.array([r["ecc"] for r in rows if r["ecc"] is not None], dtype=float)
        if len(vals) >= 2:
            mu, sd = float(vals.mean()), float(vals.std(ddof=1) or 1.0)
            for r in rows:
                if r["ecc"] is not None:
                    r["ecc_z"] = (r["ecc"] - mu) / sd
    except Exception as e:
        with open(ECC_LOG, "a", encoding="utf-8") as flog:
            flog.write(f"z-score failed: {e}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for r in rows: fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(ECC_LOG, "a", encoding="utf-8") as flog:
        flog.write(f"[{datetime.now().isoformat(timespec='seconds')}] ECC done → {out_path}\n")
    _stage_mark("ecc","done")

# -------- Report --------
def write_markdown_report(queries_file: Path, answers_file: Path, out_md: Path, max_items: int = 100):
    _banner("WRITE MARKDOWN REPORT"); _stage_mark("report","start")
    qs = {}
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line); qs[ex["id"]] = ex["query"]
    rows = []
    with open(answers_file, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            qid = ex["id"]; q = qs.get(qid,""); ans = ex.get("answer","").strip()
            bm = "NA"
            try:
                docs = ex.get("docs", [])
                if docs: bm = max((d.get("bm25",-1) for d in docs))
            except Exception: pass
            rows.append((qid, q, bm, ans))
            if len(rows) >= max_items: break
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# Bench Report ({datetime.now().isoformat(timespec='seconds')})\n\n")
        f.write(f"- **Shown**: {len(rows)} rows\n")
        f.write(f"- **Retriever**: BM25 → CE (`ms-marco-MiniLM-L6-v2`), keep top-3\n")
        f.write(f"- **LLM**: Qwen/Qwen2.5-7B-Instruct bf16 (no quant)\n\n")
        f.write("| ID | Top BM25 | Query | Answer (truncated) |\n")
        f.write("|---:|---------:|-------|---------------------|\n")
        for qid, q, bm, ans in rows:
            an = (ans[:200] + "…") if len(ans) > 200 else ans
            q_clean = q.replace("|","\\|"); an_clean = an.replace("\n"," ").replace("|","\\|")
            f.write(f"| {qid} | {bm} | {q_clean} | {an_clean} |\n")
    log.info(f"Wrote Markdown report -> {out_md}")
    _stage_mark("report","done")
def write_combined_csv(ans_file, mars_file, ecc_file, safe_file, out_csv):
    import csv
    A,B,C,D = {},{}, {},{}

    for f, target in [(mars_file,B),(ecc_file,C),(safe_file,D)]:
        if not Path(f).exists(): continue
        with open(f, "r") as x:
            for line in x:
                j = json.loads(line)
                target[j["id"]] = j

    with open(ans_file,"r") as f:
        with open(out_csv,"w",newline="") as g:
            w = csv.writer(g)
            w.writerow(["id","mars","ecc","ecc_z","safe"])
            for line in f:
                ex = json.loads(line)
                qid = ex["id"]
                mars = B.get(qid,{}).get("mars_score")
                ecc  = C.get(qid,{}).get("ecc")
                eccz = C.get(qid,{}).get("ecc_z")
                safe = D.get(qid,{}).get("safe_score")
                w.writerow([qid,mars,ecc,eccz,safe])


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    # corpus
    ap.add_argument("--shard_size", type=int, default=100_000)
    ap.add_argument("--threads", type=int, default=8)
    # retrieval + rerank
    ap.add_argument("--k_first", type=int, default=1000, help="BM25 candidates before rerank")
    ap.add_argument("--keep", type=int, default=3, help="docs kept after rerank")
    ap.add_argument("--rerank_ce_model", type=str, default="cross-encoder/ms-marco-MiniLM-L6-v2")
    ap.add_argument("--rerank_batch", type=int, default=64)
    # queries
    ap.add_argument("--k", type=int, default=100, help="(unused; kept for back-compat)")
    ap.add_argument("--query_src", type=str, required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--n_queries", type=int, default=50)
    # model
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--skip_gen", action="store_true")
    ap.add_argument("--non_strict", action="store_true")
    # UE
    ap.add_argument("--do_mars", action="store_true")
    ap.add_argument("--do_ecc", action="store_true")
    ap.add_argument("--do_safe", action="store_true")
    ap.add_argument("--ecc_embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    # outputs
    ap.add_argument("--out_prefix", type=str, default="bench50")
    args = ap.parse_args()

    _banner("FULL RUN START")
    Path("logs/start.ts").write_text(datetime.now().isoformat(timespec="seconds"))
    log_sysinfo()
    log.info(f"Args: {vars(args)}")

    # Names derived from source and seed
    src_path = Path(args.query_src); src_tag = src_path.stem
    sampled_queries = Path(f"data/queries/{args.out_prefix}.{src_tag}.seed{args.seed}.jsonl")
    retr_file = RETR_DIR / f"{args.out_prefix}.{src_tag}.seed{args.seed}.reranked.top{args.keep}.jsonl"
    ans_file  = ANS_DIR  / f"{args.out_prefix}.{src_tag}.seed{args.seed}.qwen7b_bf16.jsonl"
    mars_file = UE_DIR   / f"{args.out_prefix}.{src_tag}.seed{args.seed}.mars.jsonl"
    ecc_file  = UE_DIR   / f"{args.out_prefix}.{src_tag}.seed{args.seed}.ecc.jsonl"
    report_md = REPORTS_DIR / f"{args.out_prefix}.{src_tag}.seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    try:
        gz = download_wiki18()
        stream_full_to_shards(gz, FULL_SHARDS_DIR, shard_size=args.shard_size)

        # optional sanity
        try:
            total_lines = 0
            for p in FULL_SHARDS_DIR.glob("*.jsonl"):
                with open(p, "r", encoding="utf-8") as f:
                    for _ in f: total_lines += 1
            log.info(f"[SANITY] Full shards contain ~{total_lines:,} docs")
        except Exception as e:
            log.info(f"[SANITY] unable to count docs: {e}")

        build_bm25_index(FULL_SHARDS_DIR, FULL_INDEX_DIR, threads=args.threads)
        sample_queries_from_file(src_path, sampled_queries, n=args.n_queries, seed=args.seed)

        # NEW: BM25 + Cross-Encoder rerank
        run_retrieval_with_rerank(
            sampled_queries, FULL_INDEX_DIR, retr_file,
            k_first=args.k_first, k_keep=args.keep,
            ce_model=args.rerank_ce_model, batch_size=args.rerank_batch
        )

        if not args.skip_gen:
            generate_answers_with_resume(retr_file, ans_file, args.model,
                                         max_new_tokens=256, strict=not args.non_strict)

            # UE
        if args.do_mars:
            run_mars_whitebox(ans_file, mars_file, model_id=args.model)

        if args.do_ecc:
            run_eccentricity_blackbox(ans_file, ecc_file, embed_model=args.ecc_embed_model)

        if args.do_safe:
            from safe_runner import run_safe_scores
            safe_json = UE_DIR / f"{args.out_prefix}.{src_tag}.seed{args.seed}.safe.jsonl"
            safe_csv  = UE_DIR / f"{args.out_prefix}.{src_tag}.seed{args.seed}.safe.csv"

            run_safe_scores(ans_file, safe_json, safe_csv)

            combined = REPORTS_DIR / f"{args.out_prefix}.{src_tag}.seed{args.seed}.combined_scores.csv"
            write_combined_csv(ans_file, mars_file, ecc_file, safe_json, combined)



        write_markdown_report(sampled_queries, ans_file, report_md, max_items=args.n_queries)

    except subprocess.CalledProcessError as e:
        log.exception("Subprocess failed."); sys.exit(e.returncode)
    except Exception:
        log.exception("Pipeline failed."); sys.exit(1)
    finally:
        _banner("FULL RUN DONE")
        Path("logs/done.ts").write_text(datetime.now().isoformat(timespec="seconds"))
        _stop_hb.set(); hb.join(timeout=1)
        log.info(f"Log file: {LOG_FILE.absolute()}"); sys.stdout.flush()

if __name__ == "__main__":
    main()
