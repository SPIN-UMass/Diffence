#!/bin/bash

# Declare arrays of alpha and entropy_percentile values
alphas=(0.001 0.005 0.01 0.0005)
entropy_percentiles=(0.5 0.6 0.7 0.8 0.9 0.95 0.99)
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
  for entropy_percentile in "${entropy_percentiles[@]}"; do
    # Loop over each tag
      # echo "Running for alpha=$alpha, entropy_percentile=$entropy_percentile"
      # Run the Python script and redirect output to a file
      srun -c 2 -G 1 -p gypsum-1080ti --mem=40000 python ./model_training/train-hamp.py --scan_para 1 --entropy_percentile $entropy_percentile --alpha $alpha --config ./configs/$model/hamp.yml --isTrain 0 &

  done
done

wait
echo "All jobs submitted"
