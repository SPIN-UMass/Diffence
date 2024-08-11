#!/bin/bash

defense='undefended'
model='resnet'
gpus=1

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
python parallel_run.py --config ./configs/$model/${defense}.yml --world-size $gpus 
python parallel_run.py --config ./configs/$model/${defense}.yml --world-size $gpus --diff 1

echo "Evaluating MIAs"
mkdir -p ./results/
python dist_attack.py --config ./configs/$model/${defense}.yml --world-size $gpus &> ./results/${defense}
python dist_attack.py --config ./configs/$model/${defense}.yml --world-size $gpus  --diff 1 &> ./results/${defense}_w_Diffence

