#!/usr/bin/env bash
#SBATCH --job-name=rag-bench-fast
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=90G
#SBATCH --time=08:00:00
#SBATCH --output=/d/hpc/projects/FRI/ma76193/logs/%x_%j.out
#SBATCH --error=/d/hpc/projects/FRI/ma76193/logs/%x_%j.err

set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAVA_TOOL_OPTIONS="-Xms1g -Xmx8g"

echo "======== SLURM BOOT ========"; date; hostname

on_exit() {
  echo "[DIAG] exit code: $?"
  df -h /d/hpc/projects/FRI/ma76193 || true
  free -h || true
  nvidia-smi || true
}
trap on_exit EXIT

# Correct environment activation
export ENV_DIR=/d/hpc/projects/FRI/ma76193/miniconda3/envs/ragpy311
export PATH="$ENV_DIR/bin:$PATH"

export JAVA_HOME="$ENV_DIR"
export JVM_PATH="$JAVA_HOME/lib/jvm/lib/server/libjvm.so"
export LD_LIBRARY_PATH="$(dirname "$JVM_PATH"):${LD_LIBRARY_PATH:-}"

export HF_HOME=/d/hpc/projects/FRI/ma76193/model_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"

python -V
pip -V
nvidia-smi || true

cd /d/hpc/projects/FRI/ma76193/IR_Project/src

# FIXED: install TruthTorchLM into the correct env
$ENV_DIR/bin/pip install -q --upgrade "pip<25" "setuptools<72" wheel
$ENV_DIR/bin/pip install -q accelerate einops
$ENV_DIR/bin/pip install -q sentence-transformers "faiss-cpu>=1.7.4"
$ENV_DIR/bin/pip install -q "git+https://github.com/Ybakman/TruthTorchLM.git"

# Heartbeat
( while true; do echo "[HEARTBEAT] $(date --iso-8601=seconds)"; sleep 30; done ) &
HB_PID=$!

QUERY_SRC="data/queries/factscore_bio.jsonl"

echo "[STEP] FULL RUN"
stdbuf -oL -eL python -u pipeline_full_run.py \
  --query_src "$QUERY_SRC" \
  --n_queries 3 \
  --seed 1337 \
  --k_first 1000 --keep 3 \
  --rerank_ce_model cross-encoder/ms-marco-MiniLM-L6-v2 \
  --rerank_batch 64 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --do_mars --do_ecc --do_safe \
  --ecc_embed_model sentence-transformers/all-MiniLM-L6-v2

kill "$HB_PID" || true
echo "======== SLURM DONE ========"; date
