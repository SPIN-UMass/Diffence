#!/bin/bash
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=40000  # Requested Memory
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 8:00:00  # Job time limit
#SBATCH --partition gpu-preempt # Job time limit
#SBATCH --nodes 1 #
#SBATCH --nodelist=gpu[013-041]#
#SBATCH -o ./training_log/sbatch/slurm-%j.out  # %j = job ID


while [ $# -gt 0 ]; do
  case "$1" in
    --alpha)
      alpha="$2"
      shift 2
      ;;
    --model)
      model="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

mkdir -p ./training_log/$model/relaxloss_sub
python ./model_training/train-relaxloss.py --scan_para 1 --alpha $alpha --config ./configs/$model/relaxloss.yml  &> "./training_log/$model/relaxloss_sub/${alpha}"
