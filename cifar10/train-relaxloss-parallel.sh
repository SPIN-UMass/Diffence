#!/bin/bash

# Declare arrays of alpha and entropy_percentile values
# alphas=(0.2)
alphas=(0.0001 0.0005 0.001 0.005 0.01 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0)
model=''
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
      echo "Running for alpha=$alpha "
      # Run the Python script and redirect output to a file
      sbatch train-relaxloss-parallel-each.sh --alpha $alpha --model $model
done

wait
echo "All jobs submitted"
