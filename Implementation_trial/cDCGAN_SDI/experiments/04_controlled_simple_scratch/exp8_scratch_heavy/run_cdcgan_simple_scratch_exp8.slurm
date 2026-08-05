#!/bin/bash -l

#SBATCH --job-name=scratchHeavy
#SBATCH --account=defgengan
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=0-05:00:00
#SBATCH --output=/shared/rc/defgengan/logs/%x_%j.out
#SBATCH --error=/shared/rc/defgengan/logs/%x_%j.err

set -euo pipefail

mkdir -p /shared/rc/defgengan/logs
mkdir -p /shared/rc/defgengan/outputs

spack env activate default-ml-x86_64-25052701

python -u /shared/rc/defgengan/scripts/train_cdcgan_simple_scratch_exp8.py \
  --data-dir /shared/rc/defgengan/data/simple_scratch_v1 \
  --out-root /shared/rc/defgengan/outputs/exp8_simple_scratch_100N_700S \
  --epochs 82 \
  --batch-size 16 \
  --normal-count 100 \
  --scratch-count 700 \
  --normal-per-batch 2 \
  --scratch-per-batch 14
