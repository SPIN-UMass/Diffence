#!/bin/bash
#SBATCH -c 1  # Number of Cores per Task
#SBATCH --mem=40000  # Requested Memory
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 10:00:00  # Job time limit
#SBATCH --partition gpu-preempt # Job time limit
#SBATCH --nodes 1 #
#SBATCH -o ./training_log/sbatch/slurm-%j.out  # %j = job ID


mkdir -p ./training_log/

python ./model_training/train-hamp.py  &> ./training_log/hamp
