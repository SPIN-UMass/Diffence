#!/bin/bash

# Declare arrays of alpha and entropy_percentile values
# alphas=(0.2)
alphas=(0.0001 0.0005 0.001 0.005 0.01 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0)
model='resnet'
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

# Loop over each alpha
for alpha in "${alphas[@]}"; do
  # Loop over each entropy_percentile
      # echo "Running for alpha=$alpha "
      # Run the Python script and redirect output to a file
      srun -c 2 -G 1 -p gypsum-1080ti --mem=40000 python ./model_training/train-relaxloss.py --train_org 0 --scan_para 1 --alpha $alpha --config ./configs/$model/relaxloss.yml  &
done

wait
echo "All jobs submitted"
