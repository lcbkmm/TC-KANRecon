import sys;sys.path.append('./')

from tqdm import tqdm
import numpy
from utils.Condition_aug_dataloader import double_form_dataloader

import torch
from torch.nn import functional as F
from generative.networks.nets import AutoencoderKL
from generative.inferers import DiffusionInferer
from diffusers import DDPMScheduler

from utils.common import get_parameters, save_config_to_yaml, one_to_three
from config.diffusion.config_controlnet import Config
from os.path import join as j
from accelerate import Accelerator
from torchvision.utils import make_grid, save_image
from unet.MF_UKAN import MF_UKAN
from unet.MC_model import MC_MODEL
import os

def main():
    mask = torch.load(Config.mask_path, map_location=torch.device('cpu'))
    mask = numpy.squeeze(mask)
    cf = save_config_to_yaml(Config, Config.project_dir)
    accelerator = Accelerator(**get_parameters(Accelerator, cf))
    device = 'cuda'
    val_dataloader = double_form_dataloader(
        Config.eval_path, 
        Config.sample_size, 
        Config.eval_bc, 
        Config.mode, 
        read_channel='gray',
        mask = mask)
    attention_levels = (False, ) * len(Config.up_and_down)
    vae = AutoencoderKL(
        spatial_dims=2, 
        in_channels=Config.in_channels, 
        out_channels=Config.out_channels, 
        num_channels=Config.up_and_down, 
        latent_channels=4,
        num_res_blocks=Config.num_res_layers, 
        attention_levels = attention_levels
        )
    vae = vae.eval().to(device)
    if len(Config.vae_resume_path):
        vae.load_state_dict(torch.load(Config.vae_resume_path))

    model =MF_UKAN(
    T=1000, 
    ch=64, 
    ch_mult=[1, 2, 3, 4], 
    attn=[2], 
    num_res_blocks=2, 
    dropout=0.15).to(device)
    if len(Config.sd_resume_path):
        model.load_state_dict(torch.load(Config.sd_resume_path), strict=False)
    model = model.to(device)
    
    mc_model =MC_MODEL(
    T=1000, 
    ch=64, 
    ch_mult=[1, 2, 3, 4], 
    attn=[2], 
    num_res_blocks=2, 
    dropout=0.15).to(device)
    if len(Config.sd_resume_path):
        mc_model.load_state_dict(torch.load(Config.mc_model_path), strict=False)
    mc_model = mc_model.to(device)

    
    # scheduler = DDPMScheduler(num_train_timesteps=1000)
    # 动态裁剪策略
    scheduler = DDPMScheduler(num_train_timesteps=1000, 
                                    beta_start=Config.beta_start,
                                    beta_end=Config.beta_end,
                                    beta_schedule=Config.beta_schedule,
                                    clip_sample=Config.clip_sample,
                                    clip_sample_range=Config.initial_clip_sample_range,
                                    )

    if len(Config.log_with):
        accelerator.init_trackers('train_example')

    latent_shape = None
    # scaling_factor = 1 / torch.std(next(iter(train_dataloader))[0][1])
    scaling_factor = Config.scaling_factor
    for step, batch in enumerate(val_dataloader):
        model.eval()
        mc_model.eval()
        progress_bar = tqdm(total=len(val_dataloader))
        progress_bar.set_description(f"No. {step+1}")
        mri = batch[0].float().to(device)
        mri_mask = batch[1].float().to(device)
        pt_name = batch[2][0].split('.')[0]
        with torch.no_grad():
            latent_mri = vae.encode_stage_2_inputs(mri)
        latent_shape = list(latent_mri.shape);latent_shape[0] = Config.eval_bc
        noise = torch.randn(latent_shape).to(device)
        progress_bar_sampling = tqdm(scheduler.timesteps, total=len(scheduler.timesteps), ncols=110, position=0, leave=True)
        with torch.no_grad():
            for i, t in enumerate(progress_bar_sampling):
                t_tensor = torch.tensor([t], dtype=torch.long).to(device)
                down_block_res_samples, _ = mc_model(
                x=noise, t=t_tensor, controlnet_cond=mri_mask
                )
                noise_pred = model(
                noise,
                t=t_tensor,
                down_block_additional_residuals=down_block_res_samples
                )
                
                scheduler = DDPMScheduler(num_train_timesteps=1000, 
                            beta_start=Config.beta_start,
                            beta_end=Config.beta_end,
                            beta_schedule=Config.beta_schedule,
                            clip_sample=Config.clip_sample,
                            clip_sample_range=Config.initial_clip_sample_range + Config.clip_rate * i,
                            )
                noise = scheduler.step(model_output=noise_pred, timestep=t, sample=noise).prev_sample
        with torch.no_grad():
            image = vae.decode_stage_2_outputs(noise / scaling_factor)
        
        image, mri = one_to_three(image), one_to_three(mri),
        image = torch.cat([mri, image], dim=-1)
        image = (make_grid(image, nrow=1).unsqueeze(0)+1)/2
        log_image = {"MRI": image.clamp(0, 1)}
        save_path = j(Config.output_dir, 'image_save')
        os.makedirs(save_path, exist_ok=True)
        save_image(log_image["MRI"], j(save_path, f'{pt_name}.png'))
        # accelerator.trackers[0].log_images(log_image, epoch+1)


if __name__ == '__main__':
    main()
