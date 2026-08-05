#!/bin/bash -l

#SBATCH --job-name=cdcgan3dbg
#SBATCH --account=defgengan
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=0-05:00:00
#SBATCH --output=/shared/rc/defgengan/logs/%x_%j.out
#SBATCH --error=/shared/rc/defgengan/logs/%x_%j.err

mkdir -p /shared/rc/defgengan/logs
mkdir -p /shared/rc/defgengan/scripts
mkdir -p /shared/rc/defgengan/data
mkdir -p /shared/rc/defgengan/outputs

spack env activate default-ml-x86_64-25052701

python -u /shared/rc/defgengan/scripts/train_cdcgan_normalVspot_cluster_v4_augmented.py