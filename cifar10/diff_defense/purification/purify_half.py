from math import exp
import torch
from diff_utils import *
import tqdm
import pytorch_ssim
import torchvision


def mia_purify(x, diffusion, max_iter, mode, config):
    # From noisy initialized image to purified image
    images_list = []
    transform_raw_to_diff = raw_to_diff(config.structure.dataset)
    transform_diff_to_raw = diff_to_raw(config.structure.dataset)
    x_adv = transform_raw_to_diff(x).to(config.device.diff_device)

    torchvision.utils.save_image(x[0].cpu(), 'train_img1.png')
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
            torchvision.utils.save_image(x_pur[0].cpu(), 'train_recon1.png')
            images.append(x_pur)

    return images
