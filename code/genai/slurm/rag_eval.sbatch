#!/bin/bash -l
# RAG lab evaluation on one 4-GPU node: Qwen2.5-1.5B-Instruct on GPU 0 and Qwen2.5-7B-Instruct on GPU 1,
# each running closed_book, naive_rag and improved_rag on one split. Optionally executes the learner
# notebook on GPU 2 at the same time (third argument "notebook").
#
# usage (from the folder that holds code/ and data/):
#   mkdir -p logs
#   PY=/path/to/python HF_HOME=/path/to/hf-cache sbatch --account=YOUR_ACCOUNT code/genai/slurm/rag_eval.sbatch dev results/RUN/dev-r1
#   PY=... HF_HOME=... sbatch --account=YOUR_ACCOUNT code/genai/slurm/rag_eval.sbatch test results/RUN/test notebook
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --job-name=rag-lab
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
SPLIT=$1; RUN=$2; NB=${3:-}
: "${PY:?set PY to the python of the environment}"
: "${HF_HOME:?set HF_HOME to the model cache}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$RUN" logs
nvidia-smi --query-gpu=name,memory.total --format=csv
start=$(date +%s)
pids=()
gpu=0
for MODEL in qwen2.5-1.5b qwen2.5-7b; do
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=16 "$PY" code/genai/03_evaluate.py run --split "$SPLIT" \
      --model "$MODEL" --backend hf --run-dir "$RUN" > "$RUN/log_$MODEL.txt" 2>&1 &
  pids+=($!); gpu=$((gpu + 1))
done
if [ "$NB" = "notebook" ]; then
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 "$PY" code/genai/notebooks/execute_notebook.py \
      > "$RUN/log_notebook.txt" 2>&1 &
  pids+=($!)
fi
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "{\"elapsed_seconds_generation\": $(( $(date +%s) - start ))}" > "$RUN/job_timing.json"
grep -hE "questions in|cache hits|Error|Traceback" "$RUN"/log_*.txt || true
"$PY" code/genai/03_evaluate.py summarise --run-dir "$RUN" --out "$RUN/summary.json" || rc=1
exit $rc
