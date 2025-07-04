#!/bin/bash
#SBATCH -J varlen_bd3lm                # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=100000                  # server memory requested (per node)
#SBATCH -t 1:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu_h100          # Request partition
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

#Loading modules 
module load 2024 
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

#Environment setup
export VENV_PATH="/home/avettoruzzo/bd3lms/bd3lm_venv"      ### Refer to your desired virtual environment path

# Check that VENV_PATH is set
if [ -z "$VENV_PATH" ]; then
    echo "Error: VENV_PATH is not set. Please export the path to your virtual environment:"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

BLOCK_SIZE=4
LENGTH=2048

srun python -u main.py \
    loader.eval_batch_size=1 \
    model=small \
    algo=bd3lm \
    algo.backbone=hf_dit \
    algo.T=5000 \
    data=wikitext2 \
    model.length=$LENGTH \
    block_size=$BLOCK_SIZE \
    wandb=null \
    mode=sample_eval \
    eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE} \
    model.attn_backend=sdpa \
    sampling.nucleus_p=0.9 \
    sampling.kv_cache=true \
    sampling.logdir=$PWD/sample_logs/samples_genlen_bd3lm_blocksize${BLOCK_SIZE} \