import os
import argparse
import time
import numpy as np
# from runx.logx import logx
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
#from models import ResNet18
from attack import  AdversaryTwo_HopSkipJump,AdversaryTwo_SaltandPepperNoise, Model_with_QueryNum
# from cert_radius.certify import certify
import yaml
import sys
sys.path.append("..")
from utils import parse_config
import subprocess,shlex
from utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

action = -1
 
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

def load_saved_results(data_path, world_size):
    l0_rank_mem = []
    l0_rank_nonmem = []

    l1_rank_mem = []
    l1_rank_nonmem = []

    l2_rank_mem = []
    l2_rank_nonmem = []

    linf_rank_mem = []
    linf_rank_nonmem = []

    for rank in range(world_size):
   
        data = np.load(os.path.join(data_path, f'{world_size}_{rank}.npz'))
        l0_rank_mem.append(data['l0_rank_mem'])
        l0_rank_nonmem.append(data['l0_rank_nonmem'])
        l1_rank_mem.append(data['l1_rank_mem'])
        l1_rank_nonmem.append(data['l1_rank_nonmem'])
        l2_rank_mem.append(data['l2_rank_mem'])
        l2_rank_nonmem.append(data['l2_rank_nonmem'])
        linf_rank_mem.append(data['linf_rank_mem'])
        linf_rank_nonmem.append(data['linf_rank_nonmem'])
        
    l0_rank_mem = np.concatenate(l0_rank_mem)
    l0_rank_nonmem = np.concatenate(l0_rank_nonmem)
    l1_rank_mem = np.concatenate(l1_rank_mem)
    l1_rank_nonmem = np.concatenate(l1_rank_nonmem )
    l2_rank_mem = np.concatenate(l2_rank_mem)
    l2_rank_nonmem = np.concatenate(l2_rank_nonmem)
    linf_rank_mem = np.concatenate(linf_rank_mem)
    linf_rank_nonmem = np.concatenate(linf_rank_nonmem)

    return l0_rank_mem, l0_rank_nonmem, l1_rank_mem, l1_rank_nonmem, l2_rank_mem, l2_rank_nonmem, linf_rank_mem, linf_rank_nonmem

def load_saved_results_aug_cw(data_path, world_size):
    aug_rank_mem = []
    aug_rank_nonmem = []

    for rank in range(world_size):
   
        data = np.load(os.path.join(data_path, f'{world_size}_{rank}.npz'))
        aug_rank_mem.append(data['aug_rank_mem'])
        aug_rank_nonmem.append(data['aug_rank_nonmem'])
        
    aug_rank_mem = np.concatenate(aug_rank_mem)
    aug_rank_nonmem = np.concatenate(aug_rank_nonmem)

    return aug_rank_mem, aug_rank_nonmem

##############################
def main():

    parser = argparse.ArgumentParser(description='PyTorch Unrestricted Attack')
    parser.add_argument('--attack_config', type=str, default='./attack_configs/attack.yml')
    parser.add_argument('--config', type=str, default='../attack_configs/default.yml')
    parser.add_argument('--world-size', type=int) # number of GPUs
    parser.add_argument('--p','-p', type=str, default='gypsum-2080ti') # type of GPU (partition)
    parser.add_argument('--sbatch', action="store_true") # type of GPU (partition)
    parser.add_argument('--diff', type=int, default=0)
    args = parser.parse_args()

    attack_config = parse_config(args.attack_config)
    config = parse_config(args.config)

    stats_all_attacks = dict()
    stats_all_attacks_analysis=dict()

    if attack_config.label_only.attack == 'two' or attack_config.label_only.attack == 'all':
        if attack_config.label_only.generate_data==True:
            # AdversaryTwo(attack_config, Random_Data=False)
            print("Generate Data...")
            command = []
            device_num=args.world_size
            # num_per_world = math.ceil(attack_config['structure']['run_samples']/attack_config['structure']['bsize']/world_size)
            for i in range(device_num):
                if args.sbatch==True:
                    command.append(f'sbatch --wait -p {args.p} dist_attackTwo.sh --attack_config {args.attack_config} --rank {i} --world-size {device_num} --config {args.config} --diff {args.diff}')
                else:
                    command.append(f'srun -c 1 --gpus 1  -p {args.p} --nodes 1 --nodelist=gpu[013-041] --mem=100000 -t 8:00:00 python dist_attackTwo.py --attack_config {args.attack_config} --rank {i} --world-size {device_num} --config {args.config} --diff {args.diff}')
            # command.append('wait')
            # command.append(f'python utils/compute_data.py --world_size {world_size} --attack_config {args_.attack_config}')
            generate_workers(command)
        
        all_blackadvattack = [attack_config.label_only.blackadvattack]
        if attack_config.label_only.blackadvattack == 'all':
            all_blackadvattack = ['HopSkipJump', 'QEBA', 'GaussianNoise']

        for  blackadvattack in all_blackadvattack:
            data_path = f'./saved_data/{config.attack.target_model}/{config.attack.save_tag}/num_sample{attack_config.num_sample}/{blackadvattack}'
            l0_rank_mem, l0_rank_nonmem, l1_rank_mem, l1_rank_nonmem, l2_rank_mem, l2_rank_nonmem, linf_rank_mem, linf_rank_nonmem = load_saved_results(data_path, args.world_size)
            target_m = np.concatenate([np.ones(len(l0_rank_mem)), np.zeros(len(l0_rank_nonmem))])
            l0_stats = np.concatenate([l0_rank_mem,l0_rank_nonmem])
            l1_stats = np.concatenate([l1_rank_mem,l1_rank_nonmem])
            l2_stats = np.concatenate([l2_rank_mem,l2_rank_nonmem])
            linf_stats = np.concatenate([linf_rank_mem,linf_rank_nonmem])
            acc, auc, _, _, _, precisions_linf, recalls_linf, f1_linf, tpr_linf, fpr_linf = black_box_benchmarks._get_max_accuracy_static(target_m, linf_stats)
            stats_all_attacks[f'Boundary Attack ({blackadvattack})']=[precisions_linf, recalls_linf, f1_linf, tpr_linf, fpr_linf, acc, auc]
            stats_all_attacks_analysis[f'Boundary Attack ({blackadvattack})']=[linf_rank_mem, linf_rank_nonmem]

    if attack_config.label_only.save_result==True:
        print(stats_all_attacks)
        for key in stats_all_attacks:
            print('acc | auc', stats_all_attacks[key][-2], stats_all_attacks[key][-1] )
            print(max(stats_all_attacks[key][0]))

if __name__ == "__main__":
    main()
