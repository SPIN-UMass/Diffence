#!/bin/bash

model='resnet'

echo "Training Undefended model"
sbatch train-org.sh --model $model

echo "Training SELENA model"
python ./model_training/train-selena.py --idx_pre 1 --config ./configs/$model/selena.yml &> ./training_log/$model/selena  && #prepare data for teacher models
# sbatch train-selena.sh --train_selena 0 #prepare data for teacher models
bash train-selena-parallel.sh --model $model &&
sbatch train-selena.sh --train_selena 1 --model $model

echo "Training AdvReg model"
sbatch train-advreg.sh --model $model

echo "Training HAMP model"
# bash train-hamp-parallel.sh --model $model #scan parameters (optional)
sbatch train-hamp.sh --model $model

echo "Training Relaxloss model"
# bash train-relaxloss-parallel.sh --model $model #scan parameters (optional)
sbatch train-relaxloss.sh --model $model

echo "Training DPSGD model"
sbatch train-dpsgd.sh --model $model