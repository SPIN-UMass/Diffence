#!/bin/bash
#SBATCH -c 1  # Number of Cores per Task
#SBATCH --mem=40000  # Requested Memory
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 10:00:00  # Job time limit
#SBATCH --partition gpu-long # Job time limit
#SBATCH --nodes 1 #
#SBATCH -o ./training_log/sbatch/slurm-%j.out  # %j = job ID

model=''
while [ $# -gt 0 ]; do
  case "$1" in
    --model)
      model="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

mkdir -p ./training_log/$model
python ./model_training/train-hamp.py --config ./configs/$model/hamp.yml  &> ./training_log/$model/hamp
