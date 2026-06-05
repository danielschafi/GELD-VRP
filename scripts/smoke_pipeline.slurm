#!/bin/bash
#SBATCH --job-name=geld-smoke
#SBATCH --partition=students
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/schafhdaniel@edu.local/GELD-VRP
#SBATCH --output=/home/schafhdaniel@edu.local/GELD-VRP/result/smoke_%j.out
#SBATCH --error=/home/schafhdaniel@edu.local/GELD-VRP/result/smoke_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/home/schafhdaniel@edu.local/GELD-VRP}"
cd "$ROOT"
mkdir -p result

echo "[slurm] job_id=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown}"
echo "[slurm] cwd=$(pwd)"
echo "[slurm] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

bash scripts/smoke_pipeline.sh
