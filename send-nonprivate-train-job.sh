#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=ft-nopriv
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
#SBATCH --output=/work/fairness-privacy/slurm-logs/slurm-%A-%u.out

EPOCHS=2
BATCH_SIZE=64
MODEL_OUT="/work/fairness-privacy/models-trained/roberta-no-priv-epochs_${EPOCHS}-bs_${BATCH_SIZE}"
N_GPUS=1

torchrun --nnodes=1 --nproc-per-node=${N_GPUS} src/train.py \
    --train-mode nonprivate \
    --data-path /work/fairness-privacy/twitteraae-sentiment-data-split/ \
    --epochs $EPOCHS \
    --model-out-path $MODEL_OUT \
    --tracking-interval 10000 \
    --do-eval
    
