#!/bin/bash
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=100000  # Requested Memory
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 10:00:00  # Job time limit
#SBATCH --partition gpu-preempt # Job time limit
#SBATCH --nodes 1 #
#SBATCH --nodelist=gpu[013-041]#
#SBATCH -o ./configs/sbatch/slurm-%j.out  # %j = job ID



config=""
world_size=0
rank=0



while [ $# -gt 0 ]; do
  case "$1" in
    --attack_config)
      attack_config="$2"
      shift 2
      ;;
    --config)
      config="$2"
      shift 2
      ;;
    --world-size)
      world_size="$2"
      shift 2
      ;;
    --rank)
      rank="$2"
      shift 2
      ;;
    --diff)
      diff="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

# 运行python命令
python dist_attackTwo.py --attack_config $attack_config --config $config --world-size $world_size --rank $rank --diff $diff 