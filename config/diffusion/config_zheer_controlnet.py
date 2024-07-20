from dataclasses import dataclass


@dataclass
class Config():
    train_bc = 3
    eval_bc = 3
    num_epochs = 2000
    data_path = '/home/wangchangmiao/yuxiao/MRIRecon/mridata/firstMRI/trainImg'
    eval_path = '/home/wangchangmiao/yuxiao/MRIRecon/mridata/firstMRI/trainImg'
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
    
    # accelerate config
    split_batches = False
    mixed_precision = 'fp16'
    log_with = 'tensorboard'
    project_dir = 'weights/model/sd/exp_6_24'
