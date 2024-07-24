
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

for tag in $(seq 0 24)
do
    sbatch  train-selena-parallel-each.sh --tag $tag --model $model
done
wait;

