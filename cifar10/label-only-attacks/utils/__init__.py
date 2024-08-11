# from .architectures import *
# from .my_loader import MyCustomDataset
# from .ensemble_model import Input_diversity, MultiEnsemble
# from .util import *
from .score_based_MIA_util import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, auc
import yaml
import argparse
from opacus.dp_model_inspector import DPModelInspector
from opacus.utils import module_modification
from opacus import PrivacyEngine
from .transforms import *


def get_folder_names(path):
    folders = []
    while True:
        path, folder = os.path.split(path)

        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break

    folders.reverse()
    return folders

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

def load_data(indice=0, batch_size=128, num_samples=25000):
    transform_train = transforms.Compose([  
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=True, download=True, transform=transform_train)
    trainset=torch.utils.data.Subset(trainset, range(num_samples*indice, num_samples*(indice+1)))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)  

    return train_loader, test_loader

def load_advreg_dataset(indice=0, num_samples=25000):
    transform_train = transforms.Compose([  
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=True, download=True, transform=transform_train)
    trainset=torch.utils.data.Subset(trainset, range(num_samples*indice, num_samples*(indice+1)))
    trainset_origin = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=True, download=True, transform=transform_test)
    trainset_origin=torch.utils.data.Subset(trainset_origin, range(num_samples*indice, num_samples*(indice+1)))
    testset = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=False, download=True, transform=transform_test)
    refset = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=True, download=True, transform=transform_test)
    refset=torch.utils.data.Subset(refset, range(num_samples*(indice+1), num_samples*(indice+2)))

    return trainset,testset,refset,trainset_origin

def get_tpr(y_true, y_score, fpr_threshold=0.001, attack='None'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    accuracy = np.max(1-(fpr+(1-tpr))/2)

    auc_score = auc(fpr, tpr) 
    highest_tpr = tpr[np.where(fpr<fpr_threshold)[0][-1]]

    ## TPR and FPR are in ascending order
    ## TNR and FNR are in descending order
    fnr = 1 - tpr 
    tnr = 1 - fpr
    highest_tnr = tnr[np.where(fnr<fpr_threshold)[0][0]] 
    print( '\t\t===> %s: TPR %.2f%% @%.2f%%FPR | TNR %.2f%% @%.2f%%FNR | AUC %.4f| ACC %.4f'%( attack, highest_tpr*100, fpr_threshold*100, highest_tnr*100, fpr_threshold*100, auc_score, accuracy  ) )


def get_acc_by_ratio(y_true, y_score, ratio):
    thre = np.percentile(y_score,100-100*ratio)
    # fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # accuracy = np.max(1-(fpr+(1-tpr))/2)
    # auc_score = auc(fpr, tpr)
    print('thre:',thre)
    y_predicted = [1 if i>=thre else 0 for i in y_score]
    accuracy = np.mean(y_predicted==y_true)
    print( '\t\t===> ACC %.4f'%( accuracy ) )

    # ## TPR and FPR are in ascending order
    # ## TNR and FNR are in descending order
    # fnr = 1 - tpr 
    # tnr = 1 - fpr
    # highest_tnr = tnr[np.where(fnr<fpr_threshold)[0][0]] 
    # print( '\t\t===> %s: TPR %.2f%% @%.2f%%FPR | TNR %.2f%% @%.2f%%FNR | AUC %.4f| ACC %.4f'%( attack, highest_tpr*100, fpr_threshold*100, highest_tnr*100, fpr_threshold*100, auc_score, accuracy  ) )


class CrossEntropyLoss(nn.Module):
    """
    cross entropy loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction='none')


class MarginLoss(nn.Module):
    """
    top-5 margin loss
    """

    def __init__(self, kappa=float('inf'), k = 5):
        super().__init__()
        self.kappa = kappa
        self.k = k

    def forward(self, logits, labels, conf=1):
        onehot_label = F.one_hot(labels, num_classes=1000).float()
        true_logit5 = torch.sum(logits * onehot_label, dim=-1, keepdims=True)
        wrong_logit5, _idx = torch.topk(logits * (1-onehot_label) - onehot_label * 1e7, k=self.k, dim = 1)
        target_loss5 = torch.sum(F.relu(true_logit5 - wrong_logit5 + conf), dim = 1)
        return target_loss5

class AdvRegWarper(nn.Module):
    def __init__(self, model):
        super(AdvRegWarper, self).__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model.forward(x)
        return outputs,x