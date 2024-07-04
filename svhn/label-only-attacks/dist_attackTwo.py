import os
import argparse
import time
import random
import time
import math
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
from attack import AdversaryOne_Feature, AdversaryOne_evaluation, AdversaryTwo_HopSkipJump,AdversaryTwo_SaltandPepperNoise, Model_with_QueryNum, AdversaryTwo_QEBA, AdversaryTwo_GaussianNoise
# from cert_radius.certify import certify
import yaml
import sys
sys.path.append("..")
from utils import MyCustomDataset, get_architecture, Input_diversity, MultiEnsemble, get_dataset, get_model, load_data, parse_config

assert torch.cuda.is_available()

action = -1


sys.path.append('../util') 
sys.path.append('../') 
# sys.path.insert(0,'./util/')
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import * 
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from util.densenet import densenet
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc 
from score_based_MIA_util import black_box_benchmarks
from diff_defense.diff_warpper import * 
from tqdm import tqdm as tq
import math
from model_factory import *
os.chdir('../')


def AdversaryTwoParallel(args, config, diff_config, world_size, rank, Random_Data=False):

    num_class = config.num_classes
    if config.label_only.blackadvattack == 'HopSkipJump':
        ITER = [50] # for call HSJA evaluation [1, 5, 10, 15, 20, 30]  default 50
    elif config.label_only.blackadvattack == 'QEBA':
        ITER = [150] # for call QEBA evaluation default 150
    elif config.label_only.blackadvattack == 'SaltandPepperNoise':
        ITER = [-1] # for call SaltandPepperNoise evaluation default 150
    elif config.label_only.blackadvattack == 'GaussianNoise':
        ITER = [150] # for call SaltandPepperNoise evaluation default 150
    for maxitr in ITER:
        AUC_Dist, Distance = [], []
        torch.cuda.empty_cache()
        batch_size = 1

        cluster = config.num_sample
        # train_loader,test_loader = load_data(indice=0)
        trainset, testset, privateset,refset, trainset_origin, privateset_origin, num_classes = get_dataset()
    
        num_per_rank = math.ceil(cluster/world_size)
        all_idx = np.arange(cluster)

        rank_idx = all_idx[rank*num_per_rank: (rank+1)*num_per_rank]

        # mem_set = torch.utils.data.Subset(train_loader.dataset, rank_idx)
        # non_set = torch.utils.data.Subset(test_loader.dataset, rank_idx)
        mem_set = torch.utils.data.Subset(trainset, rank_idx)
        non_set = torch.utils.data.Subset(testset, rank_idx)

        data_set = torch.utils.data.ConcatDataset([mem_set, non_set])
        data_loader = DataLoader(data_set, batch_size=1, shuffle=False)

        # targetmodel = get_model(model_id=int(config.target_id), indice=0, model_num=0 , config=args.config).cuda().eval()
        
        num_classes=10
        resume_best= diff_config.attack.path
        targetmodel=create_model(model_name = diff_config.attack.target_model, num_classes=num_classes)
        targetmodel=targetmodel.cuda()
        if diff_config.attack.save_tag.startswith('dpsgd'):
            from opacus.dp_model_inspector import DPModelInspector
            from opacus.utils import module_modification
            from opacus import PrivacyEngine
            targetmodel = module_modification.convert_batchnorm_modules(targetmodel).to('cuda') 
        criterion=nn.CrossEntropyLoss()
        use_cuda = torch.cuda.is_available()
        assert os.path.isfile(resume_best), 'Error: no checkpoint directory %s found for best model'%resume_best
        checkpoint = os.path.dirname(resume_best)
        checkpoint = torch.load(resume_best, map_location='cuda')
        targetmodel.load_state_dict(checkpoint['state_dict'])

        if args.diff ==2:
            print('Diffence deployed')
            targetmodel =  ModelwDiff_v2_direct_mode3(targetmodel, config_path=args.diff_config)
        targetmodel = Model_with_QueryNum(targetmodel)
        


        start = time.time()
        if config.label_only.blackadvattack == 'HopSkipJump':
            AUC_Dist, Distance = AdversaryTwo_HopSkipJump(config, targetmodel, data_loader, cluster, AUC_Dist, Distance, Random_Data, maxitr)
        elif config.label_only.blackadvattack == 'QEBA':
            AUC_Dist, Distance = AdversaryTwo_QEBA(args, config, targetmodel, data_loader, cluster, AUC_Dist, Distance, Random_Data, maxitr) # need to be modified from args to config
        elif config.label_only.blackadvattack == 'SaltandPepperNoise':
            AUC_Dist, Distance = AdversaryTwo_SaltandPepperNoise(args, config, targetmodel, data_loader, cluster, AUC_Dist, Distance, Random_Data) # need to be modified from args to config
        elif config.label_only.blackadvattack == 'GaussianNoise':
            AUC_Dist, Distance = AdversaryTwo_GaussianNoise(args, config, targetmodel, data_loader, cluster, AUC_Dist, Distance, Random_Data) # need to be modified from args to config
        end = time.time()
        all_time = end - start
        print(f'inference time for each sample ({config.label_only.blackadvattack}):', all_time/len(data_loader))


    # df = pd.DataFrame()
    # print(AUC_Dist)
    # print(Distance)
    AUC_Dist = pd.DataFrame(AUC_Dist)
    Distance = pd.DataFrame(Distance)

    # print(Distance)
    l0_rank_mem = np.array(Distance.loc[:,'L0Distance'].values.tolist()).flatten()[:len(rank_idx)]
    l0_rank_nonmem = np.array(Distance.loc[:,'L0Distance'].values.tolist()).flatten()[len(rank_idx):]

    l1_rank_mem = np.array(Distance.loc[:,'L1Distance'].values.tolist()).flatten()[:len(rank_idx)]
    l1_rank_nonmem = np.array(Distance.loc[:,'L1Distance'].values.tolist()).flatten()[len(rank_idx):]

    l2_rank_mem = np.array(Distance.loc[:,'L2Distance'].values.tolist()).flatten()[:len(rank_idx)]
    l2_rank_nonmem = np.array(Distance.loc[:,'L2Distance'].values.tolist()).flatten()[len(rank_idx):]

    linf_rank_mem = np.array(Distance.loc[:,'LinfDistance'].values.tolist()).flatten()[:len(rank_idx)]
    linf_rank_nonmem = np.array(Distance.loc[:,'LinfDistance'].values.tolist()).flatten()[len(rank_idx):]
    
    data_path = f'./other-label-only-attacks/saved_data/target{config.target_id}/num_sample{config.num_sample}/{config.label_only.blackadvattack}'
    os.makedirs(data_path, exist_ok=True)
    np.savez(os.path.join(data_path,f'{world_size}_{rank}.npz'),\
                l0_rank_mem=l0_rank_mem, l0_rank_nonmem= l0_rank_nonmem, \
                    l1_rank_mem = l1_rank_mem, l1_rank_nonmem= l1_rank_nonmem, \
                        l2_rank_mem=l2_rank_mem , l2_rank_nonmem=l2_rank_nonmem,\
                             linf_rank_mem=linf_rank_mem, linf_rank_nonmem=linf_rank_nonmem)

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

##############################
def main():

    parser = argparse.ArgumentParser(description='PyTorch Unrestricted Attack')
    parser.add_argument('--config', type=str)
    parser.add_argument('--diff_config', type=str)
    parser.add_argument('--world-size', type=int)
    parser.add_argument('--rank', type=int)  
    parser.add_argument('--diff', type=int, default=0)
    args = parser.parse_args()

    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    config = parse_config(args.config[1:]) # from ../ to ./
    diff_config = parse_config(args.diff_config)


    if config.label_only.blackadvattack != 'all':
        AdversaryTwoParallel(args, config, diff_config, args.world_size, args.rank, Random_Data=False)
    else:
        config.label_only.blackadvattack = 'HopSkipJump'
        AdversaryTwoParallel(args, config,diff_config, args.world_size, args.rank, Random_Data=False)
        config.label_only.blackadvattack = 'QEBA'
        AdversaryTwoParallel(args, config, diff_config, args.world_size, args.rank, Random_Data=False)
        config.label_only.blackadvattack = 'GaussianNoise'
        AdversaryTwoParallel(args, config, diff_config, args.world_size, args.rank, Random_Data=False)

if __name__ == "__main__":
    main()
