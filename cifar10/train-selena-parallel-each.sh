#!/bin/bash
#SBATCH -c 1  # Number of Cores per Task
#SBATCH --mem=40000  # Requested Memory
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 8:00:00  # Job time limit
#SBATCH --partition gpu-preempt # Job time limit
#SBATCH --nodes 1 #
##SBATCH --nodelist=gpu[013-041]#
#SBATCH -o ./training_log/sbatch/slurm-%j.out  # %j = job ID

tag=0
model=''
while [ $# -gt 0 ]; do
  case "$1" in
    --tag)
      tag="$2"  # Capture the value next to --tag
      shift 2   # Move the argument pointer past the value
      ;;
    --model)
      model="$2" 
      shift 2 
      ;;
    *)          # If the argument doesn't match any known options, exit the loop
      break
      ;;
  esac
done

mkdir -p ./training_log/$model/selena_sub
python ./model_training/train-selena-parallel.py --train_org 1 --train_selena 0 --i $tag --config ./configs/$model/selena.yml &> ./training_log/$model/selena_sub/$tag
