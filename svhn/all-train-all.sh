#!/bin/bash


echo "Training Undefended model"
sbatch train-org.sh

echo "Training SELENA model"
python ./model_training/train-selena.py --idx_pre 1  &> ./training_log/selena  && #prepare data for teacher models
# sbatch train-selena.sh --train_selena 0 #prepare data for teacher models
bash train-selena-parallel.sh &&
sbatch train-selena.sh --train_selena 1

echo "Training AdvReg model"
sbatch train-advreg.sh 

echo "Training HAMP model"
# bash train-hamp-parallel.sh #scan parameters (optional)
sbatch train-hamp.sh 

echo "Training Relaxloss model"
# bash train-hamp-parallel.sh #scan parameters (optional)
sbatch train-relaxloss.sh

echo "Training DPSGD model"
sbatch train-dpsgd.sh