#!/bin/bash
#SBATCH -A <your_account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 4
#SBATCH --gpus-per-node=4

module load pytorch/2.6.0-1

# Auto-detect paths.
# NOTE: under sbatch, Slurm executes a COPY of this script from
# /var/spool/slurmd/job<ID>/, so ${BASH_SOURCE[0]} points at the spool copy and
# not at this model directory. Ask Slurm for the original submitted path first;
# fall back to BASH_SOURCE for `bash models/<name>/submit.sh` under salloc.
# A directory only counts if config.ini sits next to the script.
ABLATION_DIR=""
if [ -n "$SLURM_JOB_ID" ]; then
    _SUBMITTED="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
                  | sed -n 's/^[[:space:]]*Command=//p' | awk '{print $1}' | head -1)"
    if [ -n "$_SUBMITTED" ] && [ -f "$(dirname "$_SUBMITTED")/config.ini" ]; then
        ABLATION_DIR="$(cd "$(dirname "$_SUBMITTED")" && pwd)"
    fi
fi
if [ -z "$ABLATION_DIR" ]; then
    _FALLBACK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [ -f "$_FALLBACK/config.ini" ] && ABLATION_DIR="$_FALLBACK"
fi
if [ -z "$ABLATION_DIR" ]; then
    echo "FATAL: cannot locate the model directory (no config.ini found)." >&2
    echo "  tried Slurm Command=$_SUBMITTED and BASH_SOURCE=${BASH_SOURCE[0]}" >&2
    exit 1
fi
echo "model dir: $ABLATION_DIR"
ROOT_DIR="$(cd "$ABLATION_DIR/../.." && pwd)"
cd "$ROOT_DIR"

# Variable-time chain guard: this model is trained by a long `afterany` chain of
# --time-min jobs, so surplus links can fire after training is already complete.
# Bail out immediately in that case instead of burning a node-hour on data
# loading. Saves land every Nsave epochs, so the last one is at N_epoch-Nsave.
N_EPOCH_CFG=$(awk -F'=' '/^N_epoch *=/{gsub(/ /,"",$2); print $2}' "$ABLATION_DIR/config.ini")
NSAVE_CFG=$(awk -F'=' '/^Nsave *=/{gsub(/ /,"",$2); print $2}' "$ABLATION_DIR/config.ini")
LAST_SAVED=$(ls "$ABLATION_DIR/output/model/"*_model.pt 2>/dev/null \
    | sed 's|.*/||; s|_model.pt||' | sort -n | tail -1)
if [ -n "$LAST_SAVED" ] && [ -n "$N_EPOCH_CFG" ] \
   && [ "$LAST_SAVED" -ge "$((N_EPOCH_CFG - NSAVE_CFG))" ]; then
    echo "Training already complete (last saved epoch $LAST_SAVED of $N_EPOCH_CFG) — exiting."
    exit 0
fi

# set world size
NNODES=$SLURM_NNODES
echo "Detected $NNODES nodes in this job"
WORLD_SIZE=$((4 * NNODES))
echo "set world size $WORLD_SIZE"
sed -i "s/^world_size *= *.*/world_size = $WORLD_SIZE/" "$ABLATION_DIR/config.ini"

# delete all checkpoints
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

OUTPUT_DIR="$ABLATION_DIR/output"

if [ -d "$OUTPUT_DIR/model" ] && [ -f "$OUTPUT_DIR/model/latest_checkpoint.pt" ]; then
    # Resume mode
    RESUME_CKPT="$OUTPUT_DIR/model/latest_checkpoint.pt"
    sed -i "s|^output_name *=.*|output_name = $OUTPUT_DIR|" "$ABLATION_DIR/config.ini"
    sed -i "s|^load_model *=.*|load_model = True|" "$ABLATION_DIR/config.ini"
    sed -i "s|^load_model_path *=.*|load_model_path = $RESUME_CKPT|" "$ABLATION_DIR/config.ini"
    echo "Resuming from $OUTPUT_DIR (checkpoint: $RESUME_CKPT)"
else
    # Fresh start
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/inference"
    mkdir -p "$OUTPUT_DIR/model"
    mkdir -p "$OUTPUT_DIR/loss/train"
    mkdir -p "$OUTPUT_DIR/loss/validation"
    mkdir -p "$OUTPUT_DIR/test"
    mkdir -p "$OUTPUT_DIR/code"
    cp script/analyze/plot_avg.ipynb "$OUTPUT_DIR/loss" 2>/dev/null

    sed -i "s|^output_name *=.*|output_name = $OUTPUT_DIR|" "$ABLATION_DIR/config.ini"
    sed -i "s|^load_model *=.*|load_model = False|" "$ABLATION_DIR/config.ini"
    sed -i "s|^load_model_path *=.*|load_model_path = |" "$ABLATION_DIR/config.ini"

    echo "make dir $OUTPUT_DIR"

    # backup src
    echo "code backup"
    cp -r src "$OUTPUT_DIR/code"
fi

# training
echo "begin training"
srun --nodes=$SLURM_NNODES bash "$ABLATION_DIR/torchrun_script.sh" | tee "$OUTPUT_DIR/output.log"
