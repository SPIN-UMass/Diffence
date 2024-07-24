#!/bin/bash

defense='undefended'
p='gypsum-2080ti'
model='vgg13'

while [ $# -gt 0 ]; do
  case "$1" in
    --defense)
      defense="$2"
      shift 2
      ;;
    --p)
      p="$2"
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

echo "Generating outputs"
python parallel_run.py --config ./configs/$model/${defense}.yml --world-size 10 -p gpu-long --sbatch
# python parallel_run.py --config ./configs/$model/${defense}.yml --world-size 10 -p $p --diff 1 --sbatch

echo "Evaluating MIAs"
mkdir -p ./results/$model
srun -c 1 -G 1 -p gypsum-rtx8000 python dist_attack.py --config ./configs/$model/${defense}.yml --world-size 10 &> ./results/$model/${defense}
# srun -c 1 -G 1 -p gpu-preempt python dist_attack.py --config ./configs/$model/${defense}.yml --world-size 10  --diff 1 &> ./results/$model/${defense}_w_Diffence

