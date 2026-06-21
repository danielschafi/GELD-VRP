#!/bin/bash
#SBATCH --job-name=GELD-S1-Train
#SBATCH --partition=students
#SBATCH --gpus=a100:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=train-stage-1%j.log

cd /home/schafhdaniel@edu.local/GELD-VRP
uv run python -m geld_cvrptw.cli.train_stage_1 --wandb --wandb-run-name cvrptw-stage1-full-updated-lr

