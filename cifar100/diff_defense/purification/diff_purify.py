from math import exp
import torch
from ..diff_utils.transforms import *
import tqdm
from ..pytorch_ssim.ssim import ssim
import torch.nn.functional as F
from .. import pytorch_ssim
from PIL import Image
import numpy as np

def diff_purify(x, diffusion, max_iter, mode, config):
    # From noisy initialized image to purified image
    images_list = []
    transform_raw_to_diff = raw_to_diff(config.structure.dataset)
    transform_diff_to_raw = diff_to_raw(config.structure.dataset)
    x_adv = transform_raw_to_diff(x).to(config.device.diff_device)

    def cond_fn(x_reverse_t, t):
        """
        Calculate the grad of guided condition.
        """
        with torch.enable_grad():
            x_in = x_reverse_t.detach().requires_grad_(True)
            x_adv_t = diffusion.diffuse_t_steps(x_adv, t)
            # scale = exp(config.purification.guide_exp_a * t / config.purification.purify_step+config.purification.guide_exp_b) + config.purification.guide_scale_base
            if config.purification.guide_mode == 'MSE': 
                selected = -1 * F.mse_loss(x_in, x_adv_t)
                scale = diffusion.compute_scale(x_in,t, config.attack.ptb*2/255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'SSIM':
                selected = ssim(x_in, x_adv_t)
                scale = diffusion.compute_scale(x_in,t, config.attack.ptb*2/255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'CONSTANT': 
                scale = config.purification.guide_scale
            return torch.autograd.grad(selected.sum(), x_in)[0] * scale

    with torch.no_grad():
        images = []
        xt_reverse = x_adv
        for i in range(max_iter):
            # # method 1: save every step pic
            # images = []
            # xt = diffusion.diffuse_t_steps(x, config.purification.purify_step)
            # for j in range(config.purification.purify_step):
            #     xt = diffusion.denoise(xt.shape[0], n_steps=1, x=xt.to(config.device.diff_device), curr_step=(config.purification.purify_step-j), progress_bar=tqdm.tqdm)
            #     x_pur_t = xt.clone().detach()
            #     x_pur = torch.clamp(transform_diff_to_raw(x_pur_t), 0.0, 1.0)
            #     images.append(x_pur)
            # images_list.append(images)
            
            # method 2: save final step pic
            
            # xt = diffusion.diffuse_t_steps(xt_reverse, config.purification.purify_step)

            # xt_reverse = diffusion.denoise(
            #     xt.shape[0], 
            #     n_steps=config.purification.purify_step, 
            #     x=xt.to(config.device.diff_device), 
            #     curr_step=config.purification.purify_step, 
            #     # progress_bar=tqdm.tqdm,
            #     cond_fn = cond_fn if config.purification.cond else None
            # )

            purify_step = config.purification.purify_step
            xt = diffusion.diffuse_t_steps(xt_reverse, purify_step)
            xt_reverse = diffusion.denoise(
                xt.shape[0], 
                n_steps=purify_step, 
                x=xt.to(config.device.diff_device), 
                curr_step=purify_step, 
                # progress_bar=tqdm.tqdm,
                cond_fn = cond_fn if config.purification.cond else None
            )

            x_pur_t = xt_reverse.clone().detach()
            x_pur = torch.clamp(transform_diff_to_raw(x_pur_t), 0.0, 1.0)
            images.append(x_pur)

    return images

def diff_purify_v2(x, diffusion, max_iter, mode, config):
    # From noisy initialized image to purified image
    images_list = []
    transform_raw_to_diff = raw_to_diff(config.structure.dataset)
    transform_diff_to_raw = diff_to_raw(config.structure.dataset)
    x_adv = transform_raw_to_diff(x).to(config.device.diff_device)

    def cond_fn(x_reverse_t, t):
        """
        Calculate the grad of guided condition.
        """
        with torch.enable_grad():
            x_in = x_reverse_t.detach().requires_grad_(True)
            x_adv_t = diffusion.diffuse_t_steps(x_adv, t)
            # scale = exp(config.purification.guide_exp_a * t / config.purification.purify_step+config.purification.guide_exp_b) + config.purification.guide_scale_base
            if config.purification.guide_mode == 'MSE': 
                selected = -1 * F.mse_loss(x_in, x_adv_t)
                scale = diffusion.compute_scale(x_in,t, config.attack.ptb*2/255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'SSIM':
                selected = pytorch_ssim.ssim(x_in, x_adv_t)
                scale = diffusion.compute_scale(x_in,t, config.attack.ptb*2/255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'CONSTANT': 
                scale = config.purification.guide_scale
            return torch.autograd.grad(selected.sum(), x_in)[0] * scale

    with torch.no_grad():
        images = []
        xt_reverse = x_adv
        for i in range(max_iter):
            step = config.purification.purify_step
            t_steps = torch.ones(xt_reverse.shape[0], device=config.device.diff_device).long()
            t_steps = t_steps * (step-1)
            xt = diffusion.q_sample(x_0 = x_adv,t = t_steps)
            # xt = diffusion.diffuse_t_steps(xt_reverse, purify_step)
            xt_reverse = diffusion.denoise_t(xt,step)
            x_pur_t = xt_reverse.clone().detach()
            x_pur = torch.clamp(transform_diff_to_raw(x_pur_t), 0.0, 1.0)
            images.append(x_pur)

    return images


def diff_purify_v3(x, diffusion, max_iter, mode, config,FLAGS):
    #DDIM
    # From noisy initialized image to purified image

    model = diffusion.model
    

    images_list = []
    transform_raw_to_diff = raw_to_diff(config.structure.dataset)
    transform_diff_to_raw = diff_to_raw(config.structure.dataset)
    x_adv = transform_raw_to_diff(x).to(config.device.diff_device)

    def cond_fn(x_reverse_t, t):
        """
        Calculate the grad of guided condition.
        """
        with torch.enable_grad():
            x_in = x_reverse_t.detach().requires_grad_(True)
            x_adv_t = diffusion.diffuse_t_steps(x_adv, t)
            # scale = exp(config.purification.guide_exp_a * t / config.purification.purify_step+config.purification.guide_exp_b) + config.purification.guide_scale_base
            if config.purification.guide_mode == 'MSE': 
                selected = -1 * F.mse_loss(x_in, x_adv_t)
                scale = diffusion.compute_scale(x_in,t, config.attack.ptb*2/255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'SSIM':
                selected = pytorch_ssim.ssim(x_in, x_adv_t)
                scale = diffusion.compute_scale(x_in,t, config.attack.ptb*2/255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'CONSTANT': 
                scale = config.purification.guide_scale
            return torch.autograd.grad(selected.sum(), x_in)[0] * scale

    with torch.no_grad():
        images = []
        xt_reverse = x_adv
        for i in range(max_iter):
            step = config.purification.purify_step
            t_steps = torch.ones(xt_reverse.shape[0], device=config.device.diff_device).long()
            t_steps = t_steps * (step-1)
            xt = diffusion.q_sample(x_0 = x_adv,t = t_steps)
            # xt = diffusion.diffuse_t_steps(xt_reverse, purify_step)
            # xt_reverse = diffusion.denoise_t(xt,step)

            timestep=config.purification.ddim_k
            target_steps = list(range( step, -1, -timestep))
            xt_reverse = ddim_multistep(model, FLAGS, xt, step, target_steps, clip=False, device='cuda', requires_grad=False)['x_t_target']

            x_pur_t = xt_reverse.clone().detach()
            x_pur = torch.clamp(transform_diff_to_raw(x_pur_t), 0.0, 1.0)
            images.append(x_pur)
    return images



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

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))