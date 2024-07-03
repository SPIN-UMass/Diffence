#!/bin/bash
#SBATCH -c 1  # Number of Cores per Task
#SBATCH --mem=40000  # Requested Memory
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 8:00:00  # Job time limit
#SBATCH --partition gpu-preempt # Job time limit
#SBATCH --nodes 1 #
##SBATCH --nodelist=gpu[013-041]#
#SBATCH -o ./training_log/sbatch/slurm-%j.out  # %j = job ID


mkdir -p ./training_log/

python ./model_training/train-relaxloss.py --scan_para 0  &> ./training_log/relaxloss
