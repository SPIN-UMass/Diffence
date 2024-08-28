#!/bin/bash
mkdir -p ./training_log/

echo "Training Undefended model"
mkdir -p ./training_log/resnet
python ./model_training/train-org.py --config ./configs/resnet/undefended.yml &> ./training_log/resnet/undefended

echo "Training SELENA model"
mkdir -p ./training_log/resnet 
mkdir -p ./training_log/resnet/selena_sub
python ./model_training/train-selena.py --idx_pre 1 --config ./configs/resnet/selena.yml &> ./training_log/resnet/selena  && #prepare data for teacher models
for tag in $(seq 0 24)
do
    python ./model_training/train-selena-parallel.py --train_org 1 --train_selena 0 --i $tag --config ./configs/resnet/selena.yml &> ./training_log/resnet/selena_sub/$tag
done
wait;
python ./model_training/train-selena.py --train_selena 1 --config ./configs/resnet/selena.yml &> ./training_log/resnet/selena

echo "Training AdvReg model"
mkdir -p ./training_log/resnet
python ./model_training/train-advreg.py --config ./configs/resnet/advreg.yml &> ./training_log/resnet/advreg

echo "Training HAMP model"
mkdir -p ./training_log/resnet
python ./model_training/train-hamp.py --config ./configs/resnet/hamp.yml  &> ./training_log/resnet/hamp

echo "Training Relaxloss model"
mkdir -p ./training_log/resnet
python ./model_training/train-relaxloss.py --scan_para 0 --config ./configs/resnet/relaxloss.yml &> ./training_log/resnet/relaxloss

echo "Training DPSGD model"
mkdir -p ./training_log/resnet
python ./model_training/train-dpsgd.py --config ./configs/resnet/dpsgd.yml &> ./training_log/resnet/dpsgd
