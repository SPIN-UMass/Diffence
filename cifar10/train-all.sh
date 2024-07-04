#!/bin/bash
mkdir -p ./training_log/

echo "Training Undefended model"
python ./model_training/train-org.py &> ./training_log/undefended


echo "Training SELENA model"
python ./model_training/train-selena.py --idx_pre 1  &> ./training_log/selena  && #prepare data for teacher models
for tag in $(seq 0 24)
do
    python ./model_training/train-selena-parallel.py --train_org 1 --train_selena 0 --i $tag &> ./training_log/selena_sub/$tag

done
&&
python ./model_training/train-selena.py --train_selena 1  &> ./training_log/selena

echo "Training AdvReg model"
python ./model_training/train-advreg.py &> ./training_log/advreg

echo "Training HAMP model"
python ./model_training/train-hamp.py  &> ./training_log/hamp

echo "Training Relaxloss model"
python ./model_training/train-relaxloss.py --scan_para 0  &> ./training_log/relaxloss

echo "Training DPSGD model"
python ./model_training/train-dpsgd.py &> ./training_log/dpsgd
