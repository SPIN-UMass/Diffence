import os
import argparse
import pandas as pd 
import os 
import yaml
import math 
import subprocess,shlex
import re
import torchvision.transforms as transforms
import torchvision
import numpy as np
import sys
sys.path.append('../')
import torch

def generate_workers(commands):
    workers = []
    for i in range(len(commands)):
        args_list = shlex.split(commands[i])
        # stdout = open(log_files[i], "a")
        # print('executing %d-th command:\n' % i, args_list)
        p = subprocess.Popen(args_list)
        workers.append(p)

    for p in workers:
        p.wait()

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse_config(config_path=None):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
        new_config = dict2namespace(config)
    return new_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/default.yml', help='config path')
    parser.add_argument('--diff', type=int, default=0, help='config path')
    parser.add_argument('--world-size', type=int, default=0, help='config path')
    parser.add_argument('--p','-p', type=str, default='gypsum-rtx8000')
    parser.add_argument('--sbatch',action="store_true")
    args = parser.parse_args()
    # print(dict(args._get_kwargs()))

    command = []
    device_num=args.world_size
    for i in range(device_num):
        
        if args.sbatch==True:
            command.append(f'sbatch --wait -p {args.p} dist_data.sh --config {args.config} --rank {i} --world-size {device_num} --diff {args.diff}')
        else:
            command.append(f'srun -c 1 --gpus 1 --gpus-per-task 1  -p {args.p} --mem=40000 -t 8:00:00 python dist_data.py --config {args.config} --rank {i} --world-size {device_num} --diff {args.diff}')

    generate_workers(command)




