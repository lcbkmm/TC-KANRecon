from dataclasses import dataclass


@dataclass
class Config():
    train_bc = 8
    eval_bc = 8
    num_epochs = 120
    data_path = ''
    eval_path = ''
    
    single_channel = False
    mode = 'double'
    val_length = 2000

    # train process configuration
    val_inter = 2
    save_inter = 4
    sample_size = 256

    # aekl parameters
    in_channels = 1
    out_channels = 1
    up_and_down = (128, 256, 512)
    num_res_layers = 2
    scaling_factor = 0.18215
    vae_resume_path = ''


    # stable_model parameters
    sd_num_channels = (128, 256, 512, 1024)
    attention_levels = (False, False, True, True)
    sd_resume_path = ''
    controlnet_path = ''
    mc_model_path = ''
    
    # controlnet_model parameters
    conditioning_embedding_num_channels = (32, 96, 256)
    diff_loss_coefficient = 0.25
    offset_noise = False
    
    # mask
    mask_path = 'mask/mask_4.pt'
    
    # scheduler
    beta_start = 0.0008
    beta_end = 0.02
    beta_schedule = "squaredcos_cap_v2"
    clip_sample = True
    initial_clip_sample_range = 1.5
    clip_rate = 0.0021
    
    # accelerate config
    split_batches = False
    mixed_precision = 'fp16'
    log_with = 'tensorboard'
    project_dir = ''
