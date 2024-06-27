#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=ft-priv
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
#SBATCH --output=/work/fairness-privacy/slurm-logs/slurm-%A-%u.out

EPOCHS=2
BATCH_SIZE=64
EPSILON=8
MODEL_OUT="/work/fairness-privacy/models-trained/roberta-priv-eps_${EPSILON}_epochs_${EPOCHS}-bs_${BATCH_SIZE}"
N_GPUS=1

torchrun --nnodes=1 --nproc-per-node=${N_GPUS} src/train.py \
    --train-mode private \
    --data-path /work/fairness-privacy/twitteraae-sentiment-data-split/ \
    --epochs $EPOCHS \
    --model-out-path $MODEL_OUT \
    --tracking-interval 10000 \
    --priv-epsilon $EPSILON \
    --priv-max-grad-norm 0.1 \
    --do-eval
