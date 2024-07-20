import sys;sys.path.append('./')

from tqdm import tqdm
import numpy
from utils.Condition_aug_dataloader import double_form_dataloader

import torch
from torch.nn import functional as F
import argparse
from generative.networks.nets import AutoencoderKL
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet, ControlNet
from generative.networks.schedulers import DDPMScheduler

from utils.common import get_parameters, save_config_to_yaml
from config.diffusion.config_zheer_controlnet import Config
from os.path import join as j
from accelerate import Accelerator
from torchvision.utils import make_grid, save_image
from unet.Model_UKAN_Hybrid import MF_UKAN
import os

def main():
    mask = torch.load(Config.mask_path, map_location=torch.device('cpu'))
    mask = numpy.squeeze(mask)
    cf = save_config_to_yaml(Config, Config.project_dir)
    accelerator = Accelerator(**get_parameters(Accelerator, cf))
    train_dataloader = double_form_dataloader(Config.data_path, 
                                        Config.sample_size, 
                                        Config.train_bc, 
                                        Config.mode, 
                                        read_channel='gray')
    device = 'cuda'
    val_dataloader = double_form_dataloader(
        Config.eval_path, 
        Config.sample_size, 
        Config.eval_bc, 
        Config.mode, 
        read_channel='gray')
    
    device = 'cuda'
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

    scheduler = DDPMScheduler(num_train_timesteps=1000, clip_sample=True)
    # optimizer_con = torch.optim.Adam(params=controlnet.parameters(), lr=2.5e-5)
    optimizer_sd = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)
    inferer = DiffusionInferer(scheduler)

    val_interval = Config.val_inter
    save_interval = Config.save_inter

    if len(Config.log_with):
        accelerator.init_trackers('train_example')

    global_step = 0
    latent_shape = None
    # scaling_factor = 1 / torch.std(next(iter(train_dataloader))[0][1])
    scaling_factor = Config.scaling_factor
    for epoch in range(Config.num_epochs):
        model.train()
        # controlnet.train()
        epoch_loss = 0
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch+1}")
        for step, batch in enumerate(train_dataloader):
            mri = batch[0].float().to(device)
            # optimizer_con.zero_grad(set_to_none=True)
            optimizer_sd.zero_grad(set_to_none=True)
            with torch.no_grad():
                latented_mri = vae.encode_stage_2_inputs(mri)
                latented_mri = latented_mri * scaling_factor
            latent_shape = list(latented_mri.shape);latent_shape[0] = Config.train_bc
            latented_noise = torch.randn_like(latented_mri)
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (latented_mri.shape[0],), device=latented_mri.device
            ).long()
            latented_mri_noised = scheduler.add_noise(latented_mri, latented_noise, timesteps)
            # down_block_res_samples, mid_block_res_sample = controlnet(
            #     x=ffa_noised, timesteps=timesteps, controlnet_cond=slo
            # )
            noise_pred = model(
                x=latented_mri_noised,
                t=timesteps,
            )
            loss = F.mse_loss(noise_pred.float(), latented_noise.float())
            loss.backward()
            optimizer_sd.step()
            epoch_loss += loss.item()
            logs = {"loss": epoch_loss / (step + 1)}
            progress_bar.update()
            progress_bar.set_postfix(logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if (epoch + 1) % val_interval == 0 or epoch == cf['num_epochs'] - 1:
            model.eval()
            batch = next(iter(val_dataloader))
            mri = batch[0].float().to(device)
            noise = torch.randn(latent_shape).to(device)
            progress_bar_sampling = tqdm(scheduler.timesteps, total=len(scheduler.timesteps), ncols=110, position=0, leave=True)
            with torch.no_grad():
                for t in progress_bar_sampling:
                    noise_pred = model(
                    noise,
                    t=torch.Tensor((t,)).to(device),
                    )
                    noise, _ = scheduler.step(model_output=noise_pred, timestep=t, sample=noise)

            
            with torch.no_grad():
                image = vae.decode_stage_2_outputs(noise / scaling_factor)
            image = torch.cat([image, mri], dim=-1)
            image = (make_grid(image, nrow=1).unsqueeze(0)+1)/2
            log_image = {"MRI": image.clip(0, 1)}
            save_path = j(Config.project_dir, 'image_save')
            os.makedirs(save_path, exist_ok=True)
            save_image(log_image["MRI"], j(save_path, f'epoch_{epoch + 1}_firstMRI.png'))

            accelerator.trackers[0].log_images(log_image, epoch+1)

        if (epoch + 1) % save_interval == 0 or epoch == cf['num_epochs'] - 1:
            save_path = j(Config.project_dir, 'model_save')
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), j(save_path, 'model.pth'))
            # torch.save(controlnet.state_dict(), j(save_path, 'controlnet.pth'))

if __name__ == '__main__':
    main()