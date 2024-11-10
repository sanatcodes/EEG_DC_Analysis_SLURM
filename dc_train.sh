#!/bin/sh
#SBATCH --job-name=st_micro_clu
#SBATCH --output=microstate_e1%j.out
#SBATCH --error=microstate_e1%j.err
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ADAPT-CLIN
#SBATCH --partition=LARGE-G2

# Activate virtual environment
. /home/CAMPUS/d18129674/EEG_DC_Analysis_SLURM/venv


# Change to working directory
cd /home/CAMPUS/d18129674/EEG_DC_Analysis_SLURM

# Run with unbuffered output
python -u main.py
