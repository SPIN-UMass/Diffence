
for tag in $(seq 0 24)
do
    sbatch  train-selena-parallel-each.sh --tag $tag 
done
wait;

