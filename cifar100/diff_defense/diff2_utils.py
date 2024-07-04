import os 
import torch
import torchvision
import torch.nn
from torch.autograd import Variable
# from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
from absl import flags
from .diff2_model import *
# os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"

def get_FLAGS(flag_path):
    FLAGS = flags.FLAGS
    FLAGS(sys.argv[:1])
    flags.DEFINE_string('config', '', help='config path placeholder')

    flags.DEFINE_bool('train', False, help='train from scratch')
    flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
    # UNet
    flags.DEFINE_integer('ch', 128, help='base channel of UNet')
    flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
    flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
    flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
    flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
    # Gaussian Diffusion
    flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
    flags.DEFINE_float('beta_T', 0.02, help='end beta value')
    flags.DEFINE_integer('T', 1000, help='total diffusion steps')
    flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
    flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
    # Training
    flags.DEFINE_float('lr', 2e-4, help='target learning rate')
    flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
    flags.DEFINE_integer('total_steps', 800000, help='total training steps')
    flags.DEFINE_integer('img_size', 32, help='image size')
    flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
    flags.DEFINE_integer('batch_size', 128, help='batch size')
    flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
    flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
    flags.DEFINE_bool('parallel', False, help='multi gpu training')
    # Logging & Sampling
    flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
    flags.DEFINE_integer('sample_size', 64, "sampling size of images")
    flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
    # Evaluation
    flags.DEFINE_integer('save_step', 80000, help='frequency of saving checkpoints, 0 to disable during training')
    flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
    flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
    flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
    flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

    FLAGS.read_flags_from_files(flag_path)
    return FLAGS



def get_model(ckpt, FLAGS, WA=True):
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    # load model and evaluate
    ckpt = torch.load(ckpt)

    if WA:
        weights = ckpt['ema_model']
    else:
        weights = ckpt['net_model']

    new_state_dict = {}
    for key, val in weights.items():
        if key.startswith('module.'):
            new_state_dict.update({key[7:]: val})
        else:
            new_state_dict.update({key: val})

    model.load_state_dict(new_state_dict)

    model.eval()

    return model


def ddim_singlestep(model, FLAGS, x, t_c, t_target, requires_grad=False, device='cuda'):

    x = x.to(device)

    t_c = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_c)
    t_target = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_target)

    betas = torch.linspace(FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).double().to(device)
    alphas = 1. - betas
    alphas = torch.cumprod(alphas, dim=0)

    alphas_t_c = extract(alphas, t=t_c, x_shape=x.shape)
    alphas_t_target = extract(alphas, t=t_target, x_shape=x.shape)

    if requires_grad:
        epsilon = model(x, t_c)
    else:
        with torch.no_grad():
            epsilon = model(x, t_c)

    pred_x_0 = (x - ((1 - alphas_t_c).sqrt() * epsilon)) / (alphas_t_c.sqrt())
    x_t_target = alphas_t_target.sqrt() * pred_x_0 \
                 + (1 - alphas_t_target).sqrt() * epsilon

    return {
        'x_t_target': x_t_target,
        'epsilon': epsilon
    }


def ddim_multistep(model, FLAGS, x, t_c, target_steps, clip=False, device='cuda', requires_grad=False):
    for idx, t_target in enumerate(target_steps):
        result = ddim_singlestep(model, FLAGS, x, t_c, t_target, requires_grad=requires_grad, device=device)
        x = result['x_t_target']
        t_c = t_target

    if clip:
        result['x_t_target'] = torch.clip(result['x_t_target'], -1, 1)

    return result
