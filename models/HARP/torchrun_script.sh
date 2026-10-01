#!/bin/bash
ABLATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export OMP_NUM_THREADS=1
# Variable molecule sizes fragment the caching allocator; this keeps reserved
# ~= allocated so the run fits a 40 GB A100 (measured peak 29.5 / 42.4 GB).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NNODES=$SLURM_NNODES

# Per-rank stdout+stderr capture. Without this, torchrun swallows the child
# exception (error_file: <N/A>) and the piped stdout is block-buffered, so the
# real traceback is lost when a rank dies -- which is exactly what hid the
# end-of-epoch-0 failure on 16 ranks.
export PYTHONUNBUFFERED=1
export TORCHELASTIC_ERROR_FILE="$ABLATION_DIR/output/torchrun_logs/error_${SLURM_PROCID:-0}.json"
RANKLOGS="$ABLATION_DIR/output/torchrun_logs"
mkdir -p "$RANKLOGS"

torchrun --nnodes=$NNODES --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
    --log-dir="$RANKLOGS" --redirects=3 --tee=1 \
    "$ABLATION_DIR/run_train.py"
