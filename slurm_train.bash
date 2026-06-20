#!/bin/bash
#SBATCH --job-name=GELD-VRP
#SBATCH --partition=students
#SBATCH --gpus=a100:1
#SBATCH --mem=150G
#SBATCH --cpus-per-task=48
#SBATCH --time=35:00:00
#SBATCH --output=train-stage-1%j.log

#SBATCH --mail-type=end
#SBATCH --mail-user=daniel.schafi@bluewin.ch   


cd /home/schafhdaniel@edu.local/GELD-VRP
uv run python -m geld_cvrptw.cli.train_stage_1 --debug