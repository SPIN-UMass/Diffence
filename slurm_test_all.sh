#!/bin/bash

# Declare arrays of alpha and entropy_percentile values
datasets=('cifar10' 'cifar100' 'svhn')
defenses=('undefended' 'selena' 'hamp' 'relaxloss')
# defenses=('undefended' 'selena')

cur_dir=$(pwd)
# Loop over each alpha
for dataset in "${datasets[@]}"; do
  cd $cur_dir/$dataset/slurm_evaluate_MIAs/
  # Loop over each entropy_percentile
  for defense in "${defenses[@]}"; do
    # Loop over each tag
      echo "Running for dataset=$dataset, defense=$defense"
      # Run the Python script and redirect output to a file
      # bash ./$dataset/slurm_evaluate_MIAs/evaluate_mia.sh --defense $defense &
      echo "Generating outputs"
      python parallel_run.py --config ./configs/${defense}.yml --world-size 10 -p gpu-long --sbatch &
      python parallel_run.py --config ./configs/${defense}.yml --world-size 10 -p gpu-long --diff 1 --sbatch 

      echo "Evaluating MIAs"
      mkdir -p $cur_dir/results/$dataset/
      srun -c 1 -G 1 -p gypsum-2080ti --mem=100000 python dist_attack.py --config ./configs/${defense}.yml --world-size 10 &> $cur_dir/results/$dataset/${defense} &
      srun -c 1 -G 1 -p gypsum-2080ti --mem=100000 python dist_attack.py --config ./configs/${defense}.yml --world-size 10  --diff 1 &> $cur_dir/results/$dataset/${defense}_w_Diffence &
  done
done

wait
echo "All jobs submitted"
