#!/bin/bash
#SBATCH -c 1  # Number of Cores per Task
#SBATCH --mem=40000  # Requested Memory
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 72:00:00  # Job time limit
#SBATCH --partition gpu-long  # Job time limit
#SBATCH -o ./configs/sbatch/slurm-%j.out  # %j = job ID



config=""
world_size=0
rank=0



while [ $# -gt 0 ]; do
  case "$1" in
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
    *)
      break
      ;;
  esac
done

# 运行python命令
python dist_attackTwo.py --config $config --world-size $world_size --rank $rank