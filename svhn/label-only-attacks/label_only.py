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
from classifier import CNN
from dm_utils import load_dataset, init_func, Rand_Augment
from deeplearning import train_target_model, test_target_model, train_shadow_model, test_shadow_model
from attack import AdversaryOne_Feature, AdversaryOne_evaluation, AdversaryTwo_HopSkipJump,AdversaryTwo_SaltandPepperNoise, Model_with_QueryNum
# from cert_radius.certify import certify
import yaml
import sys
sys.path.append("..")
from utils import MyCustomDataset, get_architecture, Input_diversity, MultiEnsemble, get_dataset, get_model, load_data, parse_config
import subprocess,shlex
from utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

action = -1
def Train_Target_Model(config):
    shadowmodel_dir='./trained_models'
    if not os.path.exists(shadowmodel_dir):
        os.makedirs(shadowmodel_dir)
    torch.cuda.empty_cache()
    train_loader,test_loader = load_data(indice=1)
    targetmodel = get_model(model_id=int(config.target_id), indice=0, load_path=None).cuda()

    optimizer = optim.Adam(targetmodel.parameters(), lr = 0.001, betas = (0.9, 0.999), amsgrad = True, weight_decay=1e-6)
    print('======================Train_Shadow_Model====================')
    for epoch in range(1, config.epochs + 1):
        train_target_model(config, targetmodel, train_loader, optimizer, epoch)
        test_target_model(config, targetmodel, test_loader, epoch, save=True)
        if epoch % config.save_freq == 0:
                torch.save(targetmodel.state_dict(),
                        os.path.join(shadowmodel_dir,f'{config.target_id}_1_0.pt')) # can modify

def Train_Shadow_Model(config, config_path): # for transfer attack

    shadowmodel_dir='./trained_models'
    if not os.path.exists(shadowmodel_dir):
        os.makedirs(shadowmodel_dir)
    torch.cuda.empty_cache()
    train_loader,test_loader = load_data(indice=1, batch_size=config.batch_size)
    # targetmodel = get_model(model_id=int(config.target_id), indice=0, model_num=0, load_path='./trained_models' ).cuda().eval()
    targetmodel = get_model(model_id=int(config.target_id), indice=0, model_num=0 , config=config_path).cuda().eval()
    shadowmodel =  get_model(model_id=int(config.target_id), indice=1,load=False).cuda()

    optimizer = optim.Adam(shadowmodel.parameters(), lr = 0.001, betas = (0.9, 0.999), amsgrad = True, weight_decay=1e-6)
    print('======================Train_Shadow_Model====================')
    start = time.time()
    for epoch in range(1, config.epochs + 1):
        train_shadow_model(config, targetmodel, shadowmodel, train_loader, optimizer, epoch)
        test_shadow_model(config, targetmodel, shadowmodel, test_loader, epoch, save=False)
        if epoch % config.save_freq == 0:
                torch.save(shadowmodel.state_dict(),
                        os.path.join(shadowmodel_dir,f'{config.target_id}_1_-1.pt'))
    end = time.time()
    print('shadow model training time:', end-start)
   
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
    parser.add_argument('--config', type=str, default='./cifar10.yml')
    parser.add_argument('--diff_config', type=str, default='../configs/default.yml')
    parser.add_argument('--world-size', type=int) # number of GPUs
    parser.add_argument('--p','-p', type=str, default='gypsum-2080ti') # type of GPU (partition)
    parser.add_argument('--sbatch', action="store_true") # type of GPU (partition)
    parser.add_argument('--diff', type=int, default=0)
    args = parser.parse_args()

    config = parse_config(args.config)

    stats_all_attacks = dict()
    stats_all_attacks_analysis=dict()

    if config.label_only.attack == 'two' or config.label_only.attack == 'all':
        if config.label_only.generate_data==True:
            # AdversaryTwo(config, Random_Data=False)
            print("Generate Data...")
            command = []
            device_num=args.world_size
            # num_per_world = math.ceil(config['structure']['run_samples']/config['structure']['bsize']/world_size)
            for i in range(device_num):
                if args.sbatch==True:
                    command.append(f'sbatch --wait -p {args.p} dist_attackTwo.sh --config {args.config} --rank {i} --world-size {device_num} --diff_config {args.diff_config} --diff {args.diff}')
                else:
                    command.append(f'srun -c 1 --gpus 1  -p {args.p} --mem=40000 -t 8:00:00 python dist_attackTwo.py --config {args.config} --rank {i} --world-size {device_num} --diff_config {args.diff_config} --diff {args.diff}')
            # command.append('wait')
            # command.append(f'python utils/compute_data.py --world_size {world_size} --config {args_.config}')
            generate_workers(command)
        
        all_blackadvattack = [config.label_only.blackadvattack]
        if config.label_only.blackadvattack == 'all':
            all_blackadvattack = ['HopSkipJump', 'QEBA', 'GaussianNoise']

        for  blackadvattack in all_blackadvattack:
            data_path = f'./saved_data/target{config.target_id}/num_sample{config.num_sample}/{blackadvattack}'
            l0_rank_mem, l0_rank_nonmem, l1_rank_mem, l1_rank_nonmem, l2_rank_mem, l2_rank_nonmem, linf_rank_mem, linf_rank_nonmem = load_saved_results(data_path, args.world_size)
            target_m = np.concatenate([np.ones(len(l0_rank_mem)), np.zeros(len(l0_rank_nonmem))])
            l0_stats = np.concatenate([l0_rank_mem,l0_rank_nonmem])
            l1_stats = np.concatenate([l1_rank_mem,l1_rank_nonmem])
            l2_stats = np.concatenate([l2_rank_mem,l2_rank_nonmem])
            linf_stats = np.concatenate([linf_rank_mem,linf_rank_nonmem])
            acc, auc, _, _, _, precisions_linf, recalls_linf, f1_linf, tpr_linf, fpr_linf = black_box_benchmarks._get_max_accuracy_static(target_m, linf_stats)
            stats_all_attacks[f'Boundary Attack ({blackadvattack})']=[precisions_linf, recalls_linf, f1_linf, tpr_linf, fpr_linf, acc, auc]
            stats_all_attacks_analysis[f'Boundary Attack ({blackadvattack})']=[linf_rank_mem, linf_rank_nonmem]


    if config.label_only.save_result==True:
        print(stats_all_attacks)
        # plot precision-recall for all attacks
        
        plt.figure(figsize=(8,5), dpi= 100)
        for key in stats_all_attacks:
            print('acc | auc', stats_all_attacks[key][-2], stats_all_attacks[key][-1] )
            print(max(stats_all_attacks[key][0]))
            plt.plot(stats_all_attacks[key][1], stats_all_attacks[key][0], label=key, alpha=.7)

        # stats_all_attacks['Distance-CW']=[precisions_aug, recalls_aug]
        plt.scatter([0.5],[0.5], label='Random Guess', color='r')
        plt.xlabel("Recall",fontsize=15)#横坐标名字
        plt.ylabel("Precision",fontsize=15)
        plt.ylim(0,1.0)
        # plt.title(f'dist',fontsize=20)
        plt.legend(loc = "best",fontsize=15)
        plt.savefig(f'./pre_rec.png',dpi=300,bbox_inches='tight')
        print('Save all precision and recall...')
        # Save
        # np.save(f'../parallel_run_with_config/other_label_only_{config.target_id}.npy', stats_all_attacks) 
        # np.save(f'../parallel_run_with_config/other_label_only_analysis_{config.target_id}.npy', stats_all_attacks_analysis) 

if __name__ == "__main__":
    main()
