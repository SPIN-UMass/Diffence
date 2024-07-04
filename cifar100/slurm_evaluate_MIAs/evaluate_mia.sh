#!/bin/bash

defense='undefended'

while [ $# -gt 0 ]; do
  case "$1" in
    --defense)
      defense="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

echo "Generating outputs"
python parallel_run.py --config ./configs/${defense}.yml --world-size 10 -p gypsum-2080ti 
python parallel_run.py --config ./configs/${defense}.yml --world-size 10 -p gypsum-2080ti --diff 1

echo "Evaluating MIAs"
mkdir -p ./results/
srun -c 1 -G 1 -p gypsum-2080ti python dist_attack.py --config ./configs/${defense}.yml --world-size 10 &> ./results/${defense}
srun -c 1 -G 1 -p gypsum-2080ti python dist_attack.py --config ./configs/${defense}.yml --world-size 10  --diff 1 &> ./results/${defense}_w_Diffence

