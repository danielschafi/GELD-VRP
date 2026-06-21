#!/bin/bash
#SBATCH --job-name=CVRPTW-Data-Gen
#SBATCH --partition=students
#SBATCH --mem=150G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=data-gen-%j.log


cd /home/schafhdaniel@edu.local/GELD-VRP
uv run python -m scripts/generate_cvrptw.py --num-samples 20000 
