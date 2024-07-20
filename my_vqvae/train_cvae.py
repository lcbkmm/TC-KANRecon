import random
import sys

from tqdm import tqdm;sys.path.append('./')
import numpy
import torch
from torch.nn import functional as F
from my_vqvae.conditional_vae import AutoencoderKL
from my_vqvae.conditional_encoder import MaskConditionEncoder

from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator
from utils.Condition_aug_dataloader import double_form_dataloader
from utils.common import get_parameters, save_config_to_yaml
from config.vae.config_monaivae_zheer import Config
from os.path import join as j
from accelerate import Accelerator
from torchvision.utils import make_grid, save_image
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
                                        read_channel='gray',
                                        mask=mask)
    device = 'cuda'
    val_dataloader = double_form_dataloader(
        Config.eval_path, 
        Config.sample_size, 
        Config.eval_bc, 
        Config.mode, 
        read_channel='gray')
    
    up_and_down = Config.up_and_down
    attention_levels = (False, ) * len(up_and_down)
    vae = AutoencoderKL(
        spatial_dims=2, 
        in_channels=Config.in_channels, 
        out_channels=Config.out_channels, 
        num_channels=Config.up_and_down, 
        latent_channels=4,
        num_res_blocks=Config.num_res_layers, 
        attention_levels = attention_levels
        )
    if len(Config.vae_path):
        vae.load_state_dict(torch.load(Config.vae_path))

    con_encoder = MaskConditionEncoder(3, Config.up_and_down[0], 
                                       Config.up_and_down[-1], 
                                       stride=4)
    
    discriminator = PatchDiscriminator(spatial_dims=2, num_channels=64, 
                                       in_channels=Config.out_channels, out_channels=1)
    if len(Config.dis_path):
        discriminator.load_state_dict(torch.load(Config.dis_path))

    perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="vgg")
    vae, con_encoder, discriminator, perceptual_loss = vae.to(device), con_encoder.to(device), discriminator.to(device), perceptual_loss.to(device)
    vae.requires_grad_(False).eval()

    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = 0.01
    perceptual_weight = 0.001
    
    optimizer_g = torch.optim.Adam(params=con_encoder.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4)

    val_interval = Config.val_inter
    save_interval = Config.save_inter
    autoencoder_warm_up_n_epochs = 0 if (len(Config.vae_path) and len(Config.dis_path)) else Config.autoencoder_warm_up_n_epochs

    if len(Config.log_with):
        accelerator.init_trackers('train_example')

    global_step = 0
    for epoch in range(Config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}")
        con_encoder.train()
        discriminator.train()
        for step, batch in enumerate(train_dataloader):
            batch = batch[0]
            images = batch[random.randint(1,2)].to(device).clip(-1, 1)
            slo = batch[0].to(device).clip(-1, 1)
            # print("Image size: ", images.shape)
            optimizer_g.zero_grad(set_to_none=True)
            condition_im = con_encoder(slo)
            reconstruction, _, _ = vae(images, condition_im=condition_im)

            recons_loss = F.mse_loss(reconstruction.float(), images.float())

            
            p_loss = perceptual_loss(reconstruction.float(), images.float())
            loss_g = recons_loss + perceptual_weight * p_loss
            # loss_g = recons_loss + perceptual_weight * p_loss

            if epoch+1 > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            if epoch+1 > autoencoder_warm_up_n_epochs:
                optimizer_d.zero_grad(set_to_none=True)

                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = adv_weight * discriminator_loss

                loss_d.backward()
                optimizer_d.step()

            progress_bar.update(1)
            logs = {"gen_loss": loss_g.detach().item(), 
                    "dis_loss": loss_d.detach().item() if epoch+1 > autoencoder_warm_up_n_epochs else 0, 
                    "pp_loss": p_loss.detach().item(), 
                    "adv_loss": generator_loss.detach().item() if epoch+1 > autoencoder_warm_up_n_epochs else 0}
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)
            global_step += 1

        if (epoch + 1) % val_interval == 0 or epoch == cf['num_epochs'] - 1:
            con_encoder.eval()
            total_mse_loss = 0.0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    batch = batch[0]
                    slo, first_ffa, second_ffa = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    # Calculate reconstruction for the first and second part of the batch
                    condition_im = con_encoder(slo)
                    val_recon_1, _, _ = vae(first_ffa, condition_im=condition_im)
                    val_recon_2, _, _ = vae(second_ffa, condition_im=condition_im)
                    # Compute the MSE loss for each part of the batch
                    mse_loss_1 = F.mse_loss(val_recon_1, first_ffa)
                    mse_loss_2 = F.mse_loss(val_recon_2, second_ffa)
                    # Accumulate the total loss
                    total_mse_loss += (mse_loss_1 + mse_loss_2) / 2.0  # Averaging loss for both parts
                    # Optionally, you can save images here as well
                    if batch_idx == 0:  # Example: Save only the first batch for illustration
                        save_path = j(Config.project_dir, 'image_save')
                        os.makedirs(save_path, exist_ok=True)
                        val_recon = torch.cat([first_ffa, val_recon_1], dim=-1)
                        val_recon = make_grid(val_recon, nrow=1).unsqueeze(0)
                        log_image = {"Early": (val_recon + 1) / 2}
                        val_recon = torch.cat([second_ffa, val_recon_2], dim=-1)
                        val_recon = make_grid(val_recon, nrow=1).unsqueeze(0)
                        log_image["Late"] = (val_recon + 1) / 2
                        save_image(log_image["Early"].clip(0, 1), j(save_path, f'epoch_{epoch + 1}_Early.png'))
                        save_image(log_image["Late"].clip(0, 1), j(save_path, f'epoch_{epoch + 1}_Late.png'))
                        # accelerator.trackers[0].log_images(log_image, epoch+1)
                # Calculate the average MSE loss over the entire validation dataset
                average_mse_loss = total_mse_loss / len(val_dataloader)
                # Print or log the average MSE loss
                print(f'Epoch {epoch + 1}, Average MSE Loss: {average_mse_loss.item()}')
                del average_mse_loss, total_mse_loss, mse_loss_1, mse_loss_2
                
        
        if (epoch + 1) % save_interval == 0 or epoch == cf['num_epochs'] - 1:
            gen_path = j(Config.project_dir, 'gen_save')
            dis_path = j(Config.project_dir, 'dis_save')
            os.makedirs(gen_path, exist_ok=True)
            os.makedirs(dis_path, exist_ok=True)
            torch.save(vae.state_dict(), j(gen_path, 'vae.pth'))
            torch.save(discriminator.state_dict(), j(dis_path, 'dis.pth'))
            

    



if __name__ == '__main__':
    main()