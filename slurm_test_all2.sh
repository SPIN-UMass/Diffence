#!/bin/bash

# Declare arrays of datasets, defenses, Ns, Ts, and modes
datasets=('cifar10' 'cifar100' 'svhn')
defenses=('undefended')
Ns=(2 5 10 20 30 40 50 60)
Ts=(40 60 80 100 120 140 160 180 200)
modes=(1 2 3)
diff=1

# # Declare arrays of datasets, defenses, Ns, Ts, and modes
# datasets=('cifar10')
# defenses=('memguard')
# Ns=(30)
# Ts=(160)
# modes=(1 2 3)
# diff=1

model='resnet'
p='gpu-long'

cur_dir=$(pwd)
# Loop over each dataset
for dataset in "${datasets[@]}"; do
  cd $cur_dir/$dataset/slurm_evaluate_MIAs/
  # Loop over each defense
  for defense in "${defenses[@]}"; do
    # Loop over each N
    python parallel_run.py --config ./configs/$model/${defense}.yml --world-size 10 --p $p --sbatch 
    
    mkdir -p $cur_dir/results/diff$diff/$model/$dataset/
    srun -c 2 -G 1 -p gypsum-2080ti --mem=100000 python dist_attack.py --config ./configs/$model/${defense}.yml --world-size 10  &> $cur_dir/results/diff$diff/$model/$dataset/${defense} &
    for N in "${Ns[@]}"; do
      # Loop over each T
      for T in "${Ts[@]}"; do
        echo "Running for dataset=$dataset, model=$model, defense=$defense, N=$N, T=$T, diff=$diff"
        # Run the Python script and redirect output to a file
        echo "Generating outputs"
        if [ "$defense" != "memguard" ]; then
            python parallel_run.py --config ./configs/$model/${defense}.yml --world-size 10 --p $p --diff $diff --sbatch --N $N --T $T
            # Loop over each mode
            for mode in "${modes[@]}"; do
                echo "Evaluating MIAs for mode=$mode"
                mkdir -p $cur_dir/results/diff$diff/$model/N${N}_T${T}/mode${mode}/$dataset/
                srun -c 2 -G 1 -p gypsum-2080ti --mem=100000 python dist_attack.py --config ./configs/$model/${defense}.yml --world-size 10 --diff $diff --N $N --T $T --mode $mode &> $cur_dir/results/diff$diff/$model/N${N}_T${T}/mode${mode}/$dataset/${defense}_w_Diffence &
            done
        else
            # Loop over each mode
            for mode in "${modes[@]}"; do
                echo "Evaluating MIAs for mode=$mode"
                mkdir -p $cur_dir/results/diff$diff/$model/N${N}_T${T}/mode${mode}/$dataset/
                python parallel_run.py --config ./configs/$model/${defense}.yml --world-size 10 --p $p --diff $diff --sbatch --N $N --T $T --mode $mode
                srun -c 2 -G 1 -p gypsum-2080ti --mem=100000 python dist_attack.py --config ./configs/$model/${defense}.yml --world-size 10 --diff $diff --N $N --T $T --mode $mode &> $cur_dir/results/diff$diff/$model/N${N}_T${T}/mode${mode}/$dataset/${defense}_w_Diffence &
            done
        fi  
      done
    done
  done
done

wait
echo "All jobs submitted"
