from PIL import Image
import numpy as np
import torch.nn.functional as F
import torch
import pickle
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from .purification.diff_purify import *
from .pytorch_diffusion.diffusion import Diffusion
from .diff_utils.accuracy import *
from .diff_utils.transforms import *
import yaml
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
# for v2
from .diff2_utils import *
from .diffusion import *
# for v3
from .purification.purify_imagenet import *
from .guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
import pathlib

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

class ModelwDiff(nn.Module): #DM trained from scratch
    def __init__(self, model, config_path='./configs/default.yml'):
        super(ModelwDiff, self).__init__()
        self.model = model
        self.load_diff_config(config_path)

    def load_diff_config(self, config_path):
        self.config = self.parse_config(config_path)
        DATASET_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(),'./diff_models')
        # DATASET_PATH = '/work/yuefengpeng_umass_edu/yf/exp/Hamp/cifar10/diff_defense'
        ckpt = os.path.join(DATASET_PATH, self.config.diff_model.diff_path)
        flag_path = os.path.join(DATASET_PATH, self.config.diff_model.flag_path)
        FLAGS = get_FLAGS(flag_path)
        self.FLAGS=FLAGS
        self.diff_model = get_model(ckpt, FLAGS, WA=True).to(next(self.model.parameters()).device)
        self.diffusion = GaussianDiffusion(
            self.diff_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
            mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(next(self.model.parameters()).device)
        self.transform_raw_to_clf = raw_to_clf(self.config.structure.dataset)
        # print('iter:',self.config.purification.max_iter,'steps:',self.config.purification.purify_step,'path:',self.config.purification.path_number)

    def parse_config(self, config_path=None):
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
            new_config = dict2namespace(config)
        return new_config

    def forward(self, x, ground_labels=None):
        x_nat_pur_list_list = []
        transform_clf_to_raw = clf_to_raw(self.config.structure.dataset)
        output_origin = self.model(x).detach().cpu()
        predicted_label = torch.argmax(output_origin,1)
        x = transform_clf_to_raw(x)
        for j in range(self.config.purification.path_number):
            x_nat_pur_list = diff_purify_v3(
                    x, self.diffusion, 
                    self.config.purification.max_iter, 
                    mode="purification", 
                    config=self.config,
                    FLAGS=self.FLAGS
                    )
            x_nat_pur_list_list.append(x_nat_pur_list)
        nat_list_list_dict = gen_ll(x_nat_pur_list_list, self.model, self.transform_raw_to_clf, self.config)
        # outputs= output_final_step_tensor(nat_list_list_dict)

        outputs= output_final_step_tensor_v2(nat_list_list_dict,output_origin.numpy(), predicted_label,ground_labels)
        return outputs


class ModelwDiff_direct_mode3(nn.Module): # just output one randomly selected vector
    def __init__(self, model, config_path='./configs/default.yml'):
        super(ModelwDiff_direct_mode3, self).__init__()
        self.model = model
        self.load_diff_config(config_path)

    def load_diff_config(self, config_path):
        self.config = self.parse_config(config_path)
        DATASET_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(),'./diff_models')
        # DATASET_PATH = '/work/yuefengpeng_umass_edu/yf/exp/Hamp/cifar10/diff_defense'
        ckpt = os.path.join(DATASET_PATH, self.config.diff_model.diff_path)
        flag_path = os.path.join(DATASET_PATH, self.config.diff_model.flag_path)
        FLAGS = get_FLAGS(flag_path)
        self.FLAGS=FLAGS
        self.diff_model = get_model(ckpt, FLAGS, WA=True).to(next(self.model.parameters()).device)
        self.diffusion = GaussianDiffusion(
            self.diff_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
            mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(next(self.model.parameters()).device)
        self.transform_raw_to_clf = raw_to_clf(self.config.structure.dataset)
        # print('iter:',self.config.purification.max_iter,'steps:',self.config.purification.purify_step,'path:',self.config.purification.path_number)

    def parse_config(self, config_path=None):
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
            new_config = dict2namespace(config)
        return new_config

    def forward(self, x, ground_labels=None):
        x_nat_pur_list_list = []
        transform_clf_to_raw = clf_to_raw(self.config.structure.dataset)
        output_origin = self.model(x).detach().cpu()
        predicted_label = torch.argmax(output_origin,1)
        x = transform_clf_to_raw(x)
        for j in range(self.config.purification.path_number):
            x_nat_pur_list = diff_purify_v3(
                    x, self.diffusion, 
                    self.config.purification.max_iter, 
                    mode="purification", 
                    config=self.config,
                    FLAGS=self.FLAGS
                    )
            x_nat_pur_list_list.append(x_nat_pur_list)
        nat_list_list_dict = gen_ll(x_nat_pur_list_list, self.model, self.transform_raw_to_clf, self.config)
        # outputs= output_final_step_tensor(nat_list_list_dict)

        outputs= output_final_step_tensor_v2_direct_mode3(nat_list_list_dict,output_origin.numpy(), predicted_label,ground_labels)
        return outputs
    
class ModelwDiff_v2(nn.Module): #pretrained DM on ImageNet
    def __init__(self, model, config_path='./configs/default.yml'):
        super(ModelwDiff_v2, self).__init__()
        self.model = model
        self.load_diff_config(config_path)


    def load_diff_config(self, config_path):
        self.config = self.parse_config(config_path)
        DATASET_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(),'./diff_models')
        ckpt = os.path.join(DATASET_PATH, self.config.net.model_path)
        self.diff_model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(self.config.net, model_and_diffusion_defaults().keys())
        )
        self.diff_model.load_state_dict(
            torch.load(ckpt, map_location="cpu")
        )
        self.diff_model.to(next(self.model.parameters()).device)
        if self.config.net.use_fp16:
            self.diff_model.convert_to_fp16()
        self.diff_model.eval()
        self.transform_raw_to_clf = raw_to_clf(self.config.structure.dataset)
        print('iter:',self.config.purification.max_iter,'steps:',self.config.purification.purify_step,'path:',self.config.purification.path_number)

    def parse_config(self, config_path=None):
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
            new_config = dict2namespace(config)
        return new_config

    def forward(self, x, ground_labels=None):
        x_nat_pur_list_list = []
        transform_clf_to_raw = clf_to_raw(self.config.structure.dataset)
        output_origin = self.model(x).detach().cpu()
        predicted_label = torch.argmax(output_origin,1)
        x = transform_clf_to_raw(x)
        for j in range(self.config.purification.path_number):
            x_nat_pur_list = purify_imagenet(
                    x, self.diffusion, self.diff_model, 
                    self.config.purification.max_iter, 
                    mode="purification", 
                    config=self.config
                    )
            x_nat_pur_list_list.append(x_nat_pur_list)
        nat_list_list_dict = gen_ll(x_nat_pur_list_list, self.model, self.transform_raw_to_clf, self.config)
        outputs= output_final_step_tensor_v2(nat_list_list_dict,output_origin.numpy(), predicted_label, ground_labels)
        return outputs
    
class ModelwDiff_get_changed_samples(nn.Module): #only for analysis
    def __init__(self, model, config_path='./configs/default.yml'):
        super(ModelwDiff_get_changed_samples, self).__init__()
        self.model = model
        self.load_diff_config(config_path)

    def load_diff_config(self, config_path):
        self.config = self.parse_config(config_path)
        DATASET_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(),'./diff_models')
        # DATASET_PATH = '/work/yuefengpeng_umass_edu/yf/exp/Hamp/cifar10/diff_defense'
        ckpt = os.path.join(DATASET_PATH, self.config.diff_model.diff_path)
        flag_path = os.path.join(DATASET_PATH, self.config.diff_model.flag_path)
        FLAGS = get_FLAGS(flag_path)
        self.FLAGS=FLAGS
        self.diff_model = get_model(ckpt, FLAGS, WA=True).to(next(self.model.parameters()).device)
        self.diffusion = GaussianDiffusion(
            self.diff_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
            mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(next(self.model.parameters()).device)
        self.transform_raw_to_clf = raw_to_clf(self.config.structure.dataset)
        print('iter:',self.config.purification.max_iter,'steps:',self.config.purification.purify_step,'path:',self.config.purification.path_number)

    def parse_config(self, config_path=None):
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
            new_config = dict2namespace(config)
        return new_config

    def forward(self, x, ground_labels=None):
        x_nat_pur_list_list = []
        transform_clf_to_raw = clf_to_raw(self.config.structure.dataset)
        output_origin = self.model(x).detach().cpu()
        predicted_label = torch.argmax(output_origin,1)
        x = transform_clf_to_raw(x)
        for j in range(self.config.purification.path_number):
            x_nat_pur_list = diff_purify_v3(
                    x, self.diffusion, 
                    self.config.purification.max_iter, 
                    mode="purification", 
                    config=self.config,
                    FLAGS=self.FLAGS
                    )
            x_nat_pur_list_list.append(x_nat_pur_list)

        return x_nat_pur_list_list