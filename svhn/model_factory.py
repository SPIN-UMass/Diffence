import sys
import os 
import argparse
import yaml
parser = argparse.ArgumentParser() 
# parser.add_argument('--save_tag', type=str, default='0', help='current shadow model index')
# parser.add_argument('--total_models', type=int, default=1)
# parser.add_argument('--folder_tag', type=str, default='undefended')
# parser.add_argument('--res_folder', type=str, default='lira-undefended-fullMember')
# args = parser.parse_args()

from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import *
import sys
import os 
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.distributions import Categorical
import numpy as np
from util.densenet import densenet
import util.densenet_advreg as densenet_advreg
import models
from PIL import Image
#YF********************************************************************************

MODEL_NAME_MAP = {
    'resnet': models.resnet18,
    'preactresnet18': models.preactresnet18,
    'PreActResNet18': models.PreActResNet18,
    'densenet': densenet,
    'inception': models.inceptionv3,
    'vgg':models.vgg19_bn,
    'advreg-densenet':densenet_advreg.densenet,
    'resnet20':models.resnet20,
    'resnet50': models.resnet50,
    'resnet18': models.resnet18,
    'resnet18-100': models.resnet18,
    'densenet121': models.densenet121,
    'ViT-B_16': models.ViTB_16,
    'ViT': models.ViT_Base, #fast
    'ViT3': models.vit_base, 
    'ViT4': models.ViT_4
}

def create_model(model_name,state_file=None,num_classes=10):
    if model_name not in MODEL_NAME_MAP:
        raise ValueError("Model {} is invalid. Pick from {}.".format(
            model_name, sorted(MODEL_NAME_MAP.keys())))
    model_class = MODEL_NAME_MAP[model_name]
    model = model_class(num_classes=num_classes)
    if state_file is not None:
        model.load_state_dict(torch.load(state_file))
    return model

def create_optimizer_scheduler(config, model):
    if config.optimizer.optimizer=='adam':
        optimizer = optim.Adam(model.parameters(), lr = config.optimizer.lr, betas = config.optimizer.betas, amsgrad = True, weight_decay=config.optimizer.weight_decay)
    elif config.optimizer.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.optimizer.lr, momentum=config.optimizer.momentum, weight_decay=config.optimizer.weight_decay)
    
    if config.scheduler.lr_scheduler=='MultiStepLR':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones= config.scheduler.milestones, gamma=config.scheduler.gamma) #learning rate decay
    else:
        lr_scheduler=None
    
    
    return optimizer, lr_scheduler

def create_criterion(config):
    if config.trainer.criterion == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion

def softmax_by_row(logits, T = 1.0):
    mx = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp((logits - mx)/T)
    denominator = np.sum(exp, axis=-1, keepdims=True)
    return exp/denominator

# def get_dataset():
#     num_classes = 10
#     transform_train = transforms.Compose([  
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#     dataloader = datasets.CIFAR10 
#     trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
#     testset = dataloader(root='./data', train=False, download=True, transform=transform_test)
#     return trainset, testset, num_classes

class Cifardata(data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.transform = transform
        self.labels = labels

    def __getitem__(self, index):
        img =  Image.fromarray((self.data[index].transpose(1,2,0).astype(np.uint8))) 
        label = self.labels[index]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)
    
class Hampdata(Cifardata):
    def __init__(self, cifar_set, labels):
        self.data = cifar_set.data
        self.transform = cifar_set.transform
        self.labels = labels
        self.true_labels=cifar_set.labels

    def __getitem__(self, index):
        img =  Image.fromarray((self.data[index].transpose(1,2,0).astype(np.uint8))) 
        label = self.labels[index]
        img = self.transform(img)
        true_label = self.true_labels[index]
        return img, label, true_label

    def __len__(self):
        return len(self.labels)

def get_dataset(DATASET_PATH='./data', aug=False):
    
    num_classes=10
    train_num=5000
    train_data_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'tr_data.npy')) [:train_num]
    train_label_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'tr_label.npy')) [:train_num]
    train_data_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'te_data.npy'))[:train_num]
    train_label_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'te_label.npy'))[:train_num]
    train_data = np.load(os.path.join(DATASET_PATH, 'partition', 'train_data.npy'))[:train_num*2]
    train_label = np.load(os.path.join(DATASET_PATH, 'partition', 'train_label.npy'))[:train_num*2]
    all_test_data = np.load(os.path.join(DATASET_PATH, 'partition', 'all_test_data.npy'))
    all_test_label = np.load(os.path.join(DATASET_PATH, 'partition', 'all_test_label.npy'))

    if aug==False:
        transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset=Cifardata(train_data, train_label, transform_train)
    testset =Cifardata(all_test_data, all_test_label, transform_test)
    privateset=Cifardata(train_data_tr_attack, train_label_tr_attack, transform_train)
    refset =Cifardata(train_data_te_attack , train_label_te_attack, transform_train)
    trainset_origin=Cifardata(train_data, train_label, transform_test)
    privateset_origin=Cifardata(train_data_tr_attack, train_label_tr_attack, transform_test)
    return trainset, testset, privateset,refset, trainset_origin, privateset_origin, num_classes

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

def parser_opt_scheduler(model, config):
    # idea: given model and args, return the optimizer and scheduler you choose to use

    if config.trainer.optimizer == "sgd":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=config.trainer.lr,
                                    momentum=config.trainer.momentum,  # 0.9
                                    weight_decay=config.trainer.weight_decay,  # 5e-4
                                    )
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=config.trainer.lr,
                                     betas=config.trainer.betas,
                                     weight_decay=config.trainer.weight_decay,
                                     amsgrad=True)

    if config.trainer.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.trainer.T_max)
    elif config.trainer.lr_scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.trainer.milestones ,config.trainer.steplr_gamma)
    elif config.trainer.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **({
                'factor':config.trainer.ReduceLROnPlateau_factor
               } if 'ReduceLROnPlateau_factor' in config.trainer.__dict__ else {})
        )
    else:
        scheduler = None

    return optimizer, scheduler
