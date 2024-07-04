#!/bin/bash

# Declare arrays of alpha and entropy_percentile values
alphas=(0.001 0.005 0.01 0.0005)
entropy_percentiles=(0.5 0.6 0.7 0.8 0.9 0.95 0.99)
model='resnet'

# Loop over each alpha
for alpha in "${alphas[@]}"; do
  # Loop over each entropy_percentile
  for entropy_percentile in "${entropy_percentiles[@]}"; do
    # Loop over each tag
      echo "Running for alpha=$alpha, entropy_percentile=$entropy_percentile"
      # Run the Python script and redirect output to a file
      sbatch train-hamp-parallel-each.sh --entropy_percentile $entropy_percentile --alpha $alpha --model $model
  done
done

wait
echo "All jobs submitted"
